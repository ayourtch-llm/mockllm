//! Interop tests using the `async-openai` crate as a real OpenAI-compatible client.
//!
//! These tests verify that mockllm speaks the OpenAI protocol correctly enough
//! for a production client library to parse responses without errors.

use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCalls,
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequestArgs, FinishReason,
        FunctionObjectArgs,
    },
    Client,
};
use futures::StreamExt;
use mockllm::{MockLlmServer, Response};

/// Build an async-openai Client pointed at our mock server.
fn mock_client(mock: &MockLlmServer) -> Client<OpenAIConfig> {
    let config = OpenAIConfig::new()
        .with_api_base(mock.url())
        .with_api_key("test-key");
    Client::with_config(config)
}

#[tokio::test]
async fn interop_basic_text() {
    let mock = MockLlmServer::builder()
        .next(Response::text("Hello from mockllm!"))
        .build()
        .await;

    let client = mock_client(&mock);
    let request = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hi there")
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let response = client.chat().create(request).await.unwrap();

    assert_eq!(response.choices.len(), 1);
    let msg = &response.choices[0].message;
    assert_eq!(msg.content.as_deref(), Some("Hello from mockllm!"));
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.usage.unwrap().prompt_tokens, 10);
    assert_eq!(mock.request_count(), 1);
}

#[tokio::test]
async fn interop_tool_call() {
    let mock = MockLlmServer::builder()
        .next(Response::tool_call(
            "get_weather",
            serde_json::json!({"city": "Paris", "units": "celsius"}),
        ))
        .build()
        .await;

    let client = mock_client(&mock);

    let tools = vec![ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name("get_weather")
            .description("Get weather for a city")
            .parameters(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string"}
                }
            }))
            .build()
            .unwrap(),
    })];

    let request = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("What's the weather in Paris?")
                .build()
                .unwrap(),
        )])
        .tools(tools)
        .build()
        .unwrap();

    let response = client.chat().create(request).await.unwrap();

    let choice = &response.choices[0];
    assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));

    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    match &tool_calls[0] {
        ChatCompletionMessageToolCalls::Function(tc) => {
            assert_eq!(tc.function.name, "get_weather");
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap();
            assert_eq!(args["city"], "Paris");
            assert_eq!(args["units"], "celsius");
        }
        _ => panic!("Expected function tool call"),
    }
}

#[tokio::test]
async fn interop_multi_turn_tool_loop() {
    let mock = MockLlmServer::builder()
        .next(Response::tool_call(
            "read_file",
            serde_json::json!({"path": "src/main.rs"}),
        ))
        .next(Response::text("The file contains a main function that prints hello."))
        .build()
        .await;

    let client = mock_client(&mock);

    // Turn 1: send user message, get tool call back
    let req1 = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Read src/main.rs")
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let resp1 = client.chat().create(req1).await.unwrap();
    let tool_calls = resp1.choices[0].message.tool_calls.as_ref().unwrap();

    let (tool_call_id, _fn_name) = match &tool_calls[0] {
        ChatCompletionMessageToolCalls::Function(tc) => (tc.id.clone(), tc.function.name.clone()),
        _ => panic!("Expected function tool call"),
    };

    // Turn 2: send tool result back, get text response
    let req2 = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content("Read src/main.rs")
                    .build()
                    .unwrap(),
            ),
            // Include the assistant's tool call message
            ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()
                    .unwrap(),
            ),
            // Include the tool result
            ChatCompletionRequestMessage::Tool(
                ChatCompletionRequestToolMessageArgs::default()
                    .tool_call_id(&tool_call_id)
                    .content("fn main() { println!(\"hello\"); }")
                    .build()
                    .unwrap(),
            ),
        ])
        .build()
        .unwrap();

    let resp2 = client.chat().create(req2).await.unwrap();
    assert_eq!(
        resp2.choices[0].message.content.as_deref(),
        Some("The file contains a main function that prints hello.")
    );
    assert_eq!(resp2.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(mock.request_count(), 2);
}

#[tokio::test]
async fn interop_streaming_text() {
    let mock = MockLlmServer::builder()
        .next(Response::text("Hello streaming world from mockllm!"))
        .build()
        .await;

    let client = mock_client(&mock);
    let request = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hi")
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let mut stream = client.chat().create_stream(request).await.unwrap();

    let mut collected_text = String::new();
    let mut got_finish_reason = false;

    while let Some(result) = stream.next().await {
        let chunk = result.unwrap();
        for choice in &chunk.choices {
            if let Some(ref content) = choice.delta.content {
                collected_text.push_str(content);
            }
            if choice.finish_reason.is_some() {
                got_finish_reason = true;
            }
        }
    }

    assert_eq!(collected_text, "Hello streaming world from mockllm!");
    assert!(got_finish_reason);
    assert_eq!(mock.request_count(), 1);
}

#[tokio::test]
async fn interop_streaming_tool_call() {
    let mock = MockLlmServer::builder()
        .next(Response::tool_call(
            "search",
            serde_json::json!({"query": "rust async"}),
        ))
        .build()
        .await;

    let client = mock_client(&mock);
    let request = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Search for rust async")
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let mut stream = client.chat().create_stream(request).await.unwrap();

    let mut tool_name = String::new();
    let mut tool_args = String::new();
    let mut got_finish = false;

    while let Some(result) = stream.next().await {
        let chunk = result.unwrap();
        for choice in &chunk.choices {
            if let Some(ref tcs) = choice.delta.tool_calls {
                for tc in tcs {
                    if let Some(ref f) = tc.function {
                        if let Some(ref name) = f.name {
                            tool_name.push_str(name);
                        }
                        if let Some(ref args) = f.arguments {
                            tool_args.push_str(args);
                        }
                    }
                }
            }
            if choice.finish_reason.is_some() {
                got_finish = true;
            }
        }
    }

    assert_eq!(tool_name, "search");
    let args: serde_json::Value = serde_json::from_str(&tool_args).unwrap();
    assert_eq!(args["query"], "rust async");
    assert!(got_finish);
}

#[tokio::test]
async fn interop_custom_usage() {
    let mock = MockLlmServer::builder()
        .next(Response::text("Done.").with_usage(500, 100))
        .build()
        .await;

    let client = mock_client(&mock);
    let request = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Summarize")
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let response = client.chat().create(request).await.unwrap();
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 500);
    assert_eq!(usage.completion_tokens, 100);
    assert_eq!(usage.total_tokens, 600);
}

#[tokio::test]
async fn interop_multiple_tool_calls() {
    let mock = MockLlmServer::builder()
        .next(Response::tool_calls(vec![
            ("read_file", serde_json::json!({"path": "a.rs"})),
            ("read_file", serde_json::json!({"path": "b.rs"})),
        ]))
        .build()
        .await;

    let client = mock_client(&mock);
    let request = CreateChatCompletionRequestArgs::default()
        .model("mock-model")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Read both files")
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let response = client.chat().create(request).await.unwrap();
    let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);

    for (i, path) in ["a.rs", "b.rs"].iter().enumerate() {
        match &tool_calls[i] {
            ChatCompletionMessageToolCalls::Function(tc) => {
                assert_eq!(tc.function.name, "read_file");
                let args: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap();
                assert_eq!(args["path"], *path);
            }
            _ => panic!("Expected function tool call"),
        }
    }
}

#[tokio::test]
async fn interop_fallback_generation() {
    let mock = MockLlmServer::builder()
        .fallback(|req| {
            let user_msg = req["messages"]
                .as_array()
                .and_then(|msgs| msgs.last())
                .and_then(|m| m["content"].as_str())
                .unwrap_or("unknown");
            Response::text(format!("Generated response to: {}", user_msg))
        })
        .build()
        .await;

    let client = mock_client(&mock);

    for prompt in &["hello", "how are you", "goodbye"] {
        let request = CreateChatCompletionRequestArgs::default()
            .model("mock-model")
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(*prompt)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap();

        let response = client.chat().create(request).await.unwrap();
        let content = response.choices[0].message.content.as_deref().unwrap();
        assert!(
            content.contains(prompt),
            "Expected response to contain '{}', got: {}",
            prompt,
            content
        );
    }

    assert_eq!(mock.request_count(), 3);
}
