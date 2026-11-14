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

// Async — use in #[tokio::test]
#[tokio::test]
async fn test_async() {
    let mock = MockLlmServer::builder()
        .next(Response::text("Hello!"))
        .build()
        .await;
    let url = mock.url(); // http://127.0.0.1:<port>/v1
    // ... make requests ...
}

// Sync — use in regular #[test], no runtime conflicts
#[test]
fn test_sync() {
    let mock = MockLlmServer::builder()
        .next(Response::text("Hello!"))
        .start(); // creates its own runtime on a background thread
    let url = mock.url();
    // ... test with blocking HTTP client ...
    // server shuts down when `mock` is dropped
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

## Strict mode: catch requests a real server would refuse

By default the mock accepts any JSON body. With `.strict()` it validates each chat
request the way a real OpenAI-compatible server does and answers
`400 invalid_request_error` instead of a scripted response:

- `messages` is a non-empty array of objects with a known `role`;
- every assistant `tool_calls[i].function.arguments` is a **string holding valid JSON**
  (servers parse it; vLLM answers `Expecting value: line 1 column 2 (char 1)` for a
  client that rewrote old arguments into `"[stub] {...}"`);
- every `tool` message has a `tool_call_id`.

```rust
let mock = MockLlmServer::builder()
    .strict()
    .next(Response::text("ok"))
    .start();
// ... run your client ...
assert_eq!(mock.rejected_count(), 0, "a real server would have refused a request");
```

A rejected request is still recorded, and does not consume a scripted response.

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

There's also a built-in echo fallback that returns the request body as the
response text — useful for verifying serialization:

```rust
let mock = MockLlmServer::builder()
    .fallback_echo()
    .build()
    .await;
```

## Request recording

Every request to the mock server is recorded for assertions:

```rust
// How many requests were made
mock.request_count()

// Get all recorded request bodies as serde_json::Value
mock.recorded_requests()

// Get the Nth request body, or the first/last
mock.request(0)
mock.first_request()
mock.last_request()

// Get just the messages array from the Nth request
mock.request_messages(0)

// How many scripted responses haven't been consumed yet
mock.remaining_responses()

// Assert helpers (panic with clear messages)
mock.assert_request_count(2);
mock.assert_auth(0, "Bearer sk-test-key");
mock.assert_tool_choice(0, "required");
```

## Header inspection

HTTP headers are recorded alongside request bodies:

```rust
let headers = mock.request_headers(0).unwrap();
assert_eq!(headers["authorization"], "Bearer sk-test-key");

// Or use the shorthand:
mock.assert_auth(0, "Bearer sk-test-key");
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
