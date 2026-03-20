# mockllm

A deterministic mock LLM server for testing. Speaks the OpenAI-compatible
`/v1/chat/completions` API with scripted response sequences, tool calling,
and streaming (SSE).

## Why

Testing LLM-based applications against real APIs is slow, expensive, and
non-deterministic. **mockllm** gives you a lightweight HTTP server that
returns scripted responses in order — so your tests are fast, free, and
reproducible.

## Quick start

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
mockllm = { version = "0.1", path = "../mockllm" }  # or from crates.io
```

Then in your tests:

```rust
use mockllm::{MockLlmServer, Response};

#[tokio::test]
async fn test_my_llm_app() {
    let mock = MockLlmServer::builder()
        .next(Response::text("Hello! How can I help?"))
        .build()
        .await;

    // Point your OpenAI-compatible client at mock.url()
    let api_url = mock.url(); // e.g. http://127.0.0.1:54321/v1

    // ... make requests through your app ...

    assert_eq!(mock.request_count(), 1);
    assert_eq!(mock.remaining_responses(), 0);
}
```

## Response types

```rust
use mockllm::Response;

// Text-only response (finish_reason = "stop")
Response::text("The answer is 42.")

// Single tool call (finish_reason = "tool_calls")
Response::tool_call("read_file", serde_json::json!({"path": "src/main.rs"}))

// Text + tool call
Response::text_and_tool_call(
    "Let me check that file...",
    "read_file",
    serde_json::json!({"path": "config.toml"}),
)

// Multiple tool calls in one response
Response::tool_calls(vec![
    ("search_files", serde_json::json!({"pattern": "*.rs"})),
    ("list_files", serde_json::json!({"path": "src/"})),
])

// Custom token usage
Response::text("Done.").with_usage(100, 25)
```

## Testing tool-calling loops

mockllm is especially useful for testing multi-turn tool-calling loops where
the LLM calls tools and your application feeds results back:

```rust
use mockllm::{MockLlmServer, Response};

#[tokio::test]
async fn test_tool_calling_loop() {
    let mock = MockLlmServer::builder()
        // Turn 1: LLM requests a tool call
        .next(Response::tool_call("read_file", serde_json::json!({"file_path": "src/main.rs"})))
        // Turn 2: After receiving tool result, LLM responds with text
        .next(Response::text("The file contains a main function."))
        .build()
        .await;

    // ... run your chat loop against mock.url() ...

    // Verify the interaction
    assert_eq!(mock.request_count(), 2);

    // Inspect what your app sent in the second request
    let messages = mock.request_messages(1);
    let tool_result = messages.iter().find(|m| m["role"] == "tool");
    assert!(tool_result.is_some(), "Should have sent tool result back");
}
```

## Fallback generation

When scripted responses run out, you can provide a fallback closure that
generates responses dynamically. This is the integration point for custom
generators (e.g. a character-level language model like
[hashmem](https://github.com/ayourtch/hashmem)):

```rust
use mockllm::{MockLlmServer, Response};

#[tokio::main]
async fn main() {
    let mock = MockLlmServer::builder()
        // Scripted responses are served first
        .next(Response::text("Hello! I'm a mock LLM."))
        // When those run out, the fallback kicks in
        .fallback(|request_body| {
            let last_msg = request_body["messages"]
                .as_array()
                .and_then(|msgs| msgs.last())
                .and_then(|m| m["content"].as_str())
                .unwrap_or("");
            // Generate a response however you like —
            // call into hashmem, return a canned string, etc.
            Response::text(format!("You said: {}", last_msg))
        })
        .build()
        .await;

    // First request gets "Hello! I'm a mock LLM."
    // All subsequent requests hit the fallback closure
}
```

Without a fallback, exhausted requests return a default
`"[mockllm] No more scripted responses."` message.

## Request recording

Every request to the mock server is recorded for assertions:

```rust
// How many requests were made
mock.request_count()

// Get all recorded request bodies as serde_json::Value
mock.recorded_requests()

// Get the Nth request body
mock.request(0)

// Get just the messages array from the Nth request
mock.request_messages(0)

// How many scripted responses haven't been consumed yet
mock.remaining_responses()
```

## Streaming (SSE)

Streaming works automatically. When a client sends `"stream": true`, the
server responds with `text/event-stream` SSE chunks in the standard
OpenAI format, including the final `data: [DONE]` marker.

## Standalone server

Run the included example to get a standalone mock server for manual testing
or use with external tools:

```sh
cargo run --example mock_server
cargo run --example mock_server -- --port 8080
```

```sh
curl -s http://127.0.0.1:3456/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hello"}]}'
```

## Endpoints

| Method | Path                     | Description                        |
|--------|--------------------------|------------------------------------|
| POST   | `/v1/chat/completions`   | Chat completions (streaming & non) |
| GET    | `/v1/models`             | Returns minimal model list         |

## License

MIT OR Apache-2.0
