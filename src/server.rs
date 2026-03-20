//! HTTP server implementing OpenAI-compatible `/v1/chat/completions`.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response as HyperResponse, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use tokio::net::TcpListener;

use crate::Response;

/// A function that generates a fallback [`Response`] when scripted responses are exhausted.
///
/// Receives the full request body as a [`serde_json::Value`].
pub type FallbackFn = dyn Fn(&Value) -> Response + Send + Sync;

/// A recorded HTTP request (body + headers).
struct RecordedRequest {
    body: Value,
    headers: HashMap<String, String>,
}

/// Shared state across all request handlers.
struct ServerState {
    /// Scripted responses, consumed in order.
    responses: Vec<Response>,
    /// Index of the next response to return.
    next_index: usize,
    /// All received requests (body + headers, for assertions).
    recorded_requests: Vec<RecordedRequest>,
    /// Optional fallback generator for when scripted responses run out.
    fallback: Option<Arc<FallbackFn>>,
}

/// A mock LLM HTTP server for testing.
///
/// Speaks the OpenAI-compatible `/v1/chat/completions` endpoint.
/// Returns scripted responses in sequence. Records all requests
/// for post-test assertions.
pub struct MockLlmServer {
    /// Base URL including `/v1` path, e.g. `http://127.0.0.1:12345/v1`
    base_url: String,
    /// The port the server is listening on.
    port: u16,
    /// Shared state (responses + request log).
    state: Arc<Mutex<ServerState>>,
    /// Handle to shut down the server.
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    /// Background runtime thread (only set when created via `start()`).
    _runtime_thread: Option<std::thread::JoinHandle<()>>,
}

impl MockLlmServer {
    /// Create a builder for configuring the mock server.
    pub fn builder() -> MockLlmServerBuilder {
        MockLlmServerBuilder {
            responses: Vec::new(),
            port: None,
            fallback: None,
        }
    }

    /// Get the base URL for the mock server (includes `/v1`).
    ///
    /// Pass this to your LLM client as the API URL.
    pub fn url(&self) -> String {
        self.base_url.clone()
    }

    /// Get the port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// How many requests have been received so far.
    pub fn request_count(&self) -> usize {
        self.state.lock().unwrap().recorded_requests.len()
    }

    /// Get all recorded request bodies.
    pub fn recorded_requests(&self) -> Vec<Value> {
        self.state.lock().unwrap().recorded_requests.iter().map(|r| r.body.clone()).collect()
    }

    /// Get the Nth recorded request body.
    pub fn request(&self, index: usize) -> Option<Value> {
        self.state.lock().unwrap().recorded_requests.get(index).map(|r| r.body.clone())
    }

    /// Get the first recorded request body. Panics with a clear message if none.
    pub fn first_request(&self) -> Value {
        self.request(0).expect("mockllm: no requests recorded")
    }

    /// Get the last recorded request body. Panics with a clear message if none.
    pub fn last_request(&self) -> Value {
        let count = self.request_count();
        assert!(count > 0, "mockllm: no requests recorded");
        self.request(count - 1).unwrap()
    }

    /// Assert exactly N requests were made. Panics with details if not.
    pub fn assert_request_count(&self, expected: usize) {
        let actual = self.request_count();
        assert_eq!(actual, expected, "mockllm: expected {expected} requests, got {actual}");
    }

    /// Get the messages from the Nth request.
    pub fn request_messages(&self, index: usize) -> Vec<Value> {
        self.request(index)
            .and_then(|r| r.get("messages").cloned())
            .and_then(|m| m.as_array().cloned())
            .unwrap_or_default()
    }

    /// Get recorded HTTP headers for the Nth request.
    pub fn request_headers(&self, index: usize) -> Option<HashMap<String, String>> {
        self.state.lock().unwrap().recorded_requests.get(index).map(|r| r.headers.clone())
    }

    /// Assert the Nth request had a specific Authorization header value.
    pub fn assert_auth(&self, index: usize, expected: &str) {
        let headers = self.request_headers(index).expect("mockllm: no request at index");
        let auth = headers.get("authorization").expect("mockllm: no authorization header");
        assert_eq!(auth, expected, "mockllm: authorization header mismatch");
    }

    /// Assert the Nth request used a specific `tool_choice` value.
    pub fn assert_tool_choice(&self, index: usize, expected: &str) {
        let req = self.request(index).expect("mockllm: no request at index");
        assert_eq!(
            req["tool_choice"], expected,
            "mockllm: expected tool_choice=\"{expected}\", got {:?}",
            req["tool_choice"]
        );
    }

    /// How many scripted responses remain unconsumed.
    pub fn remaining_responses(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.responses.len().saturating_sub(state.next_index)
    }

    /// Shut down the server.
    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for MockLlmServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        // Unpark the background thread so it can exit cleanly
        if let Some(thread) = self._runtime_thread.take() {
            thread.thread().unpark();
        }
    }
}

/// Builder for [`MockLlmServer`].
pub struct MockLlmServerBuilder {
    responses: Vec<Response>,
    port: Option<u16>,
    fallback: Option<Arc<FallbackFn>>,
}

impl MockLlmServerBuilder {
    /// Add the next scripted response.
    ///
    /// Responses are returned in the order they are added.
    pub fn next(mut self, response: Response) -> Self {
        self.responses.push(response);
        self
    }

    /// Set a specific port to listen on (default: random).
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set a fallback response generator for when scripted responses are exhausted.
    ///
    /// The closure receives the full request body and returns a [`Response`].
    /// Without a fallback, exhausted requests get a default
    /// `"[mockllm] No more scripted responses."` message.
    ///
    /// ```rust,no_run
    /// # use mockllm::{MockLlmServer, Response};
    /// # async fn example() {
    /// let mock = MockLlmServer::builder()
    ///     .next(Response::text("Scripted greeting"))
    ///     .fallback(|request_body| {
    ///         // Extract the last user message as a seed
    ///         let messages = request_body["messages"].as_array();
    ///         Response::text("Generated fallback response")
    ///     })
    ///     .build()
    ///     .await;
    /// # }
    /// ```
    pub fn fallback(mut self, f: impl Fn(&Value) -> Response + Send + Sync + 'static) -> Self {
        self.fallback = Some(Arc::new(f));
        self
    }

    /// Set a fallback that echoes back the request body as the response text.
    ///
    /// Useful for verifying that your client serializes requests correctly
    /// without needing to script specific responses.
    pub fn fallback_echo(self) -> Self {
        self.fallback(|req| {
            let text = serde_json::to_string_pretty(req).unwrap_or_default();
            Response::text(text)
        })
    }

    /// Build and start the mock server synchronously.
    ///
    /// Creates its own tokio runtime on a background thread. The server
    /// runs until the returned [`MockLlmServer`] is dropped.
    ///
    /// Use this instead of [`build()`](Self::build) when you need mockllm
    /// in a regular `#[test]` (not `#[tokio::test]`), or when your code
    /// under test creates its own async runtime.
    ///
    /// ```rust
    /// use mockllm::{MockLlmServer, Response};
    ///
    /// let mock = MockLlmServer::builder()
    ///     .next(Response::text("Hello"))
    ///     .start();
    ///
    /// let url = mock.url();
    /// // ... test with any HTTP client ...
    /// // server shuts down when `mock` is dropped
    /// ```
    pub fn start(self) -> MockLlmServer {
        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("mockllm: failed to create runtime");
            let mut mock = std::mem::ManuallyDrop::new(rt.block_on(self.build()));
            let state = Arc::clone(&mock.state);
            let base_url = mock.base_url.clone();
            let port = mock.port;
            let shutdown_tx = mock.shutdown_tx.take();
            tx.send((base_url, port, state, shutdown_tx)).unwrap();
            // Park the thread — the runtime stays alive, keeping the server running.
            // When MockLlmServer is dropped, it sends shutdown and this thread can exit.
            std::thread::park();
        });
        let (base_url, port, state, shutdown_tx) = rx.recv().unwrap();
        MockLlmServer {
            base_url,
            port,
            state,
            shutdown_tx,
            _runtime_thread: Some(thread),
        }
    }

    /// Build and start the mock server (async).
    ///
    /// Binds to `127.0.0.1:0` (random port) unless a port was specified.
    pub async fn build(self) -> MockLlmServer {
        let state = Arc::new(Mutex::new(ServerState {
            responses: self.responses,
            next_index: 0,
            recorded_requests: Vec::new(),
            fallback: self.fallback,
        }));

        let bind_addr = format!("127.0.0.1:{}", self.port.unwrap_or(0));
        let listener = TcpListener::bind(&bind_addr).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let port = addr.port();
        let base_url = format!("http://127.0.0.1:{}/v1", port);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server_state = Arc::clone(&state);

        tokio::spawn(async move {
            let shutdown = async { shutdown_rx.await.ok(); };
            tokio::pin!(shutdown);

            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        match accepted {
                            Ok((stream, _)) => {
                                let state = Arc::clone(&server_state);
                                let io = TokioIo::new(stream);
                                tokio::spawn(async move {
                                    let svc = service_fn(move |req| {
                                        handle_request(req, Arc::clone(&state))
                                    });
                                    if let Err(e) = http1::Builder::new()
                                        .serve_connection(io, svc)
                                        .await
                                    {
                                        eprintln!("mockllm: connection error: {}", e);
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("mockllm: accept error: {}", e);
                            }
                        }
                    }
                    _ = &mut shutdown => {
                        break;
                    }
                }
            }
        });

        MockLlmServer {
            base_url,
            port,
            state,
            shutdown_tx: Some(shutdown_tx),
            _runtime_thread: None,
        }
    }
}

/// Handle a single HTTP request.
async fn handle_request(
    req: Request<hyper::body::Incoming>,
    state: Arc<Mutex<ServerState>>,
) -> Result<HyperResponse<http_body_util::Either<Full<Bytes>, StreamBody<futures_stream::ResponseStream>>>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // GET /v1/models — minimal model list
    if method == Method::GET && (path == "/v1/models" || path == "/models") {
        let body = serde_json::json!({
            "object": "list",
            "data": [{
                "id": "mock-model",
                "object": "model",
                "owned_by": "mockllm",
            }]
        });
        return Ok(json_response(StatusCode::OK, &body));
    }

    // POST /v1/chat/completions
    if method == Method::POST
        && (path == "/v1/chat/completions" || path == "/chat/completions")
    {
        // Capture headers before consuming the request body
        let headers: HashMap<String, String> = req
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        // Read body
        let body_bytes = match req.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(e) => {
                let err = serde_json::json!({"error": {"message": format!("Failed to read body: {}", e)}});
                return Ok(json_response(StatusCode::BAD_REQUEST, &err));
            }
        };

        let body_json: Value = match serde_json::from_slice(&body_bytes) {
            Ok(v) => v,
            Err(e) => {
                let err = serde_json::json!({"error": {"message": format!("Invalid JSON: {}", e)}});
                return Ok(json_response(StatusCode::BAD_REQUEST, &err));
            }
        };

        let is_streaming = body_json.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

        // Record request and get next response
        let (response_json, sse_chunks) = {
            let mut st = state.lock().unwrap();
            let request_body = body_json.clone();
            st.recorded_requests.push(RecordedRequest { body: body_json, headers });
            let idx = st.next_index;

            if idx < st.responses.len() {
                let resp = &st.responses[idx];
                let json = resp.to_json(idx);
                let chunks = if is_streaming {
                    Some(resp.to_sse_chunks(idx))
                } else {
                    None
                };
                st.next_index += 1;
                (json, chunks)
            } else {
                // Exhausted — use fallback if set, otherwise default message
                let fallback_resp = match &st.fallback {
                    Some(f) => f(&request_body),
                    None => Response::text("[mockllm] No more scripted responses."),
                };
                let json = fallback_resp.to_json(idx);
                let chunks = if is_streaming {
                    Some(fallback_resp.to_sse_chunks(idx))
                } else {
                    None
                };
                (json, chunks)
            }
        };

        if let Some(chunks) = sse_chunks {
            // Streaming SSE response
            return Ok(sse_response(chunks));
        } else {
            // Non-streaming JSON response
            return Ok(json_response(StatusCode::OK, &response_json));
        }
    }

    // 404 for anything else
    let err = serde_json::json!({"error": {"message": format!("Not found: {} {}", method, path)}});
    Ok(json_response(StatusCode::NOT_FOUND, &err))
}

/// Build a JSON HTTP response.
fn json_response(
    status: StatusCode,
    body: &Value,
) -> HyperResponse<http_body_util::Either<Full<Bytes>, StreamBody<futures_stream::ResponseStream>>> {
    let json = serde_json::to_string(body).unwrap();
    HyperResponse::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(http_body_util::Either::Left(Full::new(Bytes::from(json))))
        .unwrap()
}

/// Build an SSE streaming response.
fn sse_response(
    chunks: Vec<String>,
) -> HyperResponse<http_body_util::Either<Full<Bytes>, StreamBody<futures_stream::ResponseStream>>> {
    let stream = futures_stream::ResponseStream::new(chunks);
    let body = StreamBody::new(stream);
    HyperResponse::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(http_body_util::Either::Right(body))
        .unwrap()
}

/// Simple stream adapter for SSE chunks.
mod futures_stream {
    use bytes::Bytes;
    use hyper::body::Frame;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    pub struct ResponseStream {
        chunks: Vec<String>,
        index: usize,
    }

    impl ResponseStream {
        pub fn new(chunks: Vec<String>) -> Self {
            Self { chunks, index: 0 }
        }
    }

    impl futures::Stream for ResponseStream {
        type Item = Result<Frame<Bytes>, std::convert::Infallible>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            if self.index < self.chunks.len() {
                let chunk = self.chunks[self.index].clone();
                self.index += 1;
                Poll::Ready(Some(Ok(Frame::data(Bytes::from(chunk)))))
            } else {
                Poll::Ready(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Response;

    #[tokio::test]
    async fn test_basic_text_response() {
        let mock = MockLlmServer::builder()
            .next(Response::text("Hello world"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();

        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert_eq!(content, "Hello world");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");

        assert_eq!(mock.request_count(), 1);
        let msgs = mock.request_messages(0);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[tokio::test]
    async fn test_tool_call_response() {
        let mock = MockLlmServer::builder()
            .next(Response::tool_call(
                "read_file",
                serde_json::json!({"file_path": "src/main.rs"}),
            ))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "read the file"}]
            }))
            .send()
            .await
            .unwrap();

        let body: Value = resp.json().await.unwrap();
        let tc = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "read_file");
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["file_path"], "src/main.rs");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn test_sequence_of_responses() {
        let mock = MockLlmServer::builder()
            .next(Response::tool_call("search", serde_json::json!({"q": "rust"})))
            .next(Response::text("Found 42 results."))
            .build()
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", mock.url());

        // First call — tool call
        let r1: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "search"}]}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(r1["choices"][0]["message"]["tool_calls"].is_array());

        // Second call — text
        let r2: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "ok"}]}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(r2["choices"][0]["message"]["content"], "Found 42 results.");
        assert_eq!(r2["choices"][0]["finish_reason"], "stop");

        assert_eq!(mock.request_count(), 2);
        assert_eq!(mock.remaining_responses(), 0);
    }

    #[tokio::test]
    async fn test_exhausted_responses() {
        let mock = MockLlmServer::builder()
            .next(Response::text("only one"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", mock.url());

        // First call — scripted
        let _: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap().json().await.unwrap();

        // Second call — fallback
        let r2: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap().json().await.unwrap();
        let content = r2["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("No more scripted"));
    }

    #[tokio::test]
    async fn test_streaming_response() {
        let mock = MockLlmServer::builder()
            .next(Response::text("Hello streaming world!"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        let body = resp.text().await.unwrap();
        assert!(body.contains("Hello streaming"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_models_endpoint() {
        let mock = MockLlmServer::builder().build().await;

        let client = reqwest::Client::new();
        let resp: Value = client
            .get(&format!("{}/models", mock.url()))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(resp["data"][0]["id"], "mock-model");
    }

    #[tokio::test]
    async fn test_fallback_called_when_exhausted() {
        let mock = MockLlmServer::builder()
            .next(Response::text("scripted"))
            .fallback(|req| {
                // Echo back the last user message
                let msg = req["messages"]
                    .as_array()
                    .and_then(|msgs| msgs.last())
                    .and_then(|m| m["content"].as_str())
                    .unwrap_or("no message");
                Response::text(format!("fallback: {}", msg))
            })
            .build()
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", mock.url());

        // First call — scripted
        let r1: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "hello"}]}))
            .send().await.unwrap().json().await.unwrap();
        assert_eq!(r1["choices"][0]["message"]["content"], "scripted");

        // Second call — hits fallback
        let r2: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "world"}]}))
            .send().await.unwrap().json().await.unwrap();
        assert_eq!(r2["choices"][0]["message"]["content"], "fallback: world");

        // Third call — fallback again with different input
        let r3: Value = client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "again"}]}))
            .send().await.unwrap().json().await.unwrap();
        assert_eq!(r3["choices"][0]["message"]["content"], "fallback: again");

        assert_eq!(mock.request_count(), 3);
    }

    #[tokio::test]
    async fn test_fallback_not_called_while_scripted_remain() {
        let fallback_called = Arc::new(Mutex::new(false));
        let flag = Arc::clone(&fallback_called);

        let mock = MockLlmServer::builder()
            .next(Response::text("first"))
            .next(Response::text("second"))
            .fallback(move |_| {
                *flag.lock().unwrap() = true;
                Response::text("should not appear")
            })
            .build()
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", mock.url());

        let r1: Value = client.post(&url)
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap().json().await.unwrap();
        assert_eq!(r1["choices"][0]["message"]["content"], "first");
        assert!(!*fallback_called.lock().unwrap());

        let r2: Value = client.post(&url)
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap().json().await.unwrap();
        assert_eq!(r2["choices"][0]["message"]["content"], "second");
        assert!(!*fallback_called.lock().unwrap());
    }

    #[tokio::test]
    async fn test_text_and_tool_call_response() {
        let mock = MockLlmServer::builder()
            .next(Response::text_and_tool_call(
                "Let me read that file for you.",
                "read_file",
                serde_json::json!({"path": "main.rs"}),
            ))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "read main.rs"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let msg = &resp["choices"][0]["message"];
        assert_eq!(msg["content"], "Let me read that file for you.");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "read_file");
        let args: Value =
            serde_json::from_str(msg["tool_calls"][0]["function"]["arguments"].as_str().unwrap())
                .unwrap();
        assert_eq!(args["path"], "main.rs");
        assert_eq!(resp["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn test_multiple_tool_calls() {
        let mock = MockLlmServer::builder()
            .next(Response::tool_calls(vec![
                ("read_file", serde_json::json!({"path": "a.rs"})),
                ("write_file", serde_json::json!({"path": "b.rs", "content": "hello"})),
            ]))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "do stuff"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let tcs = resp["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0]["function"]["name"], "read_file");
        assert_eq!(tcs[1]["function"]["name"], "write_file");
        assert_eq!(resp["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn test_with_usage() {
        let mock = MockLlmServer::builder()
            .next(Response::text("hi").with_usage(100, 50))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(resp["usage"]["prompt_tokens"], 100);
        assert_eq!(resp["usage"]["completion_tokens"], 50);
        assert_eq!(resp["usage"]["total_tokens"], 150);
    }

    #[tokio::test]
    async fn test_streaming_tool_call() {
        let mock = MockLlmServer::builder()
            .next(Response::tool_call(
                "get_weather",
                serde_json::json!({"city": "Paris"}),
            ))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        let body = resp.text().await.unwrap();

        // Should contain tool call name and arguments in SSE chunks
        assert!(body.contains("get_weather"));
        assert!(body.contains("Paris"));
        assert!(body.contains("tool_calls"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_recorded_requests() {
        let mock = MockLlmServer::builder()
            .next(Response::text("a"))
            .next(Response::text("b"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", mock.url());

        client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "first"}]}))
            .send().await.unwrap();
        client
            .post(&url)
            .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": "second"}]}))
            .send().await.unwrap();

        let recorded = mock.recorded_requests();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0]["messages"][0]["content"], "first");
        assert_eq!(recorded[1]["messages"][0]["content"], "second");

        // Also test request() accessor
        let req0 = mock.request(0).unwrap();
        assert_eq!(req0["messages"][0]["content"], "first");
        assert!(mock.request(99).is_none());
    }

    #[tokio::test]
    async fn test_invalid_json_body() {
        let mock = MockLlmServer::builder()
            .next(Response::text("unused"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .header("content-type", "application/json")
            .body("this is not valid json{{{")
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 400);
        let body: Value = resp.json().await.unwrap();
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Invalid JSON"));
    }

    #[tokio::test]
    async fn test_streaming_fallback() {
        let mock = MockLlmServer::builder()
            .fallback(|_req| Response::text("streamed fallback"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        let body = resp.text().await.unwrap();
        assert!(body.contains("streamed fallback"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_specific_port_binding() {
        // Find an available port by binding to 0 first, then releasing
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let mock = MockLlmServer::builder()
            .port(port)
            .next(Response::text("on specific port"))
            .build()
            .await;

        assert_eq!(mock.port(), port);
        assert!(mock.url().contains(&port.to_string()));

        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(resp["choices"][0]["message"]["content"], "on specific port");
    }

    #[tokio::test]
    async fn test_404_unknown_path() {
        let mock = MockLlmServer::builder().build().await;

        let client = reqwest::Client::new();

        // GET unknown path
        let resp = client
            .get(&format!("{}/unknown/endpoint", mock.url()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let body: Value = resp.json().await.unwrap();
        assert!(body["error"]["message"].as_str().unwrap().contains("Not found"));

        // POST unknown path
        let resp = client
            .post(&format!("{}/completions", mock.url()))
            .json(&serde_json::json!({"model": "t"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
    }

    #[tokio::test]
    async fn test_explicit_shutdown() {
        let mock = MockLlmServer::builder()
            .next(Response::text("before shutdown"))
            .build()
            .await;

        let port = mock.port();
        let url = mock.url();

        // Verify server is running
        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("{}/chat/completions", &url))
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap().json().await.unwrap();
        assert_eq!(resp["choices"][0]["message"]["content"], "before shutdown");

        // Explicitly shut down (consumes self, exercises the shutdown() method)
        mock.shutdown();

        // Verify the shutdown was called — the server should eventually stop.
        // We use raw TCP to avoid reqwest connection pooling.
        let mut refused = false;
        for _ in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            match tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port)).await {
                Err(_) => {
                    refused = true;
                    break;
                }
                Ok(_) => continue,
            }
        }
        assert!(refused, "Port should stop accepting connections after shutdown");
    }

    #[tokio::test]
    async fn test_body_read_failure() {
        // Test the body read error path by sending a request with Content-Length
        // that doesn't match the body, causing hyper to fail reading.
        let mock = MockLlmServer::builder()
            .next(Response::text("unused"))
            .build()
            .await;

        let port = mock.port();

        // Use raw TCP to send a malformed HTTP request with truncated body
        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port))
            .await
            .unwrap();

        // Send headers claiming 1000 bytes of body, then close the connection
        let request = b"POST /v1/chat/completions HTTP/1.1\r\n\
                        Host: 127.0.0.1\r\n\
                        Content-Type: application/json\r\n\
                        Content-Length: 1000\r\n\
                        \r\n\
                        {\"partial\": true}";
        stream.write_all(request).await.unwrap();
        // Close write side immediately — server will fail to read the full body
        drop(stream);

        // Give the server a moment to process the error
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Server should still be running and handle subsequent requests
        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("http://127.0.0.1:{}/v1/chat/completions", port))
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(resp["choices"][0]["message"]["content"], "unused");
    }

    #[tokio::test]
    async fn test_connection_error_malformed_http() {
        // Triggers hyper's serve_connection error (line 203) by sending
        // garbage that isn't valid HTTP at all.
        let mock = MockLlmServer::builder()
            .next(Response::text("still works"))
            .build()
            .await;

        let port = mock.port();

        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port))
            .await
            .unwrap();

        // Send binary garbage — not valid HTTP
        stream.write_all(b"\x00\x01\x02GARBAGE_NOT_HTTP\xff\xfe").await.unwrap();
        // Shut down the write side so hyper sees EOF
        stream.shutdown().await.unwrap();
        drop(stream);

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Server should still be healthy
        let client = reqwest::Client::new();
        let resp: Value = client
            .post(&format!("http://127.0.0.1:{}/v1/chat/completions", port))
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(resp["choices"][0]["message"]["content"], "still works");
    }

    #[tokio::test]
    async fn test_streaming_text_and_tool_call() {
        let mock = MockLlmServer::builder()
            .next(Response::text_and_tool_call(
                "Checking the file...",
                "read_file",
                serde_json::json!({"path": "main.rs"}),
            ))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "read it"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        let body = resp.text().await.unwrap();

        // Should contain both text content and tool call data
        assert!(body.contains("Checking the file"));
        assert!(body.contains("read_file"));
        assert!(body.contains("main.rs"));
        assert!(body.contains("tool_calls"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_multiple_tool_calls() {
        let mock = MockLlmServer::builder()
            .next(Response::tool_calls(vec![
                ("search", serde_json::json!({"q": "foo"})),
                ("list", serde_json::json!({"dir": "/tmp"})),
            ]))
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "do both"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();

        let body = resp.text().await.unwrap();
        assert!(body.contains("search"));
        assert!(body.contains("list"));
        assert!(body.contains("foo"));
        assert!(body.contains("/tmp"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_first_and_last_request() {
        let mock = MockLlmServer::builder()
            .next(Response::text("a"))
            .next(Response::text("b"))
            .next(Response::text("c"))
            .build()
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", mock.url());
        for msg in &["first", "second", "third"] {
            client.post(&url)
                .json(&serde_json::json!({"model": "t", "messages": [{"role": "user", "content": msg}]}))
                .send().await.unwrap();
        }

        assert_eq!(mock.first_request()["messages"][0]["content"], "first");
        assert_eq!(mock.last_request()["messages"][0]["content"], "third");
    }

    #[tokio::test]
    async fn test_assert_request_count() {
        let mock = MockLlmServer::builder()
            .next(Response::text("a"))
            .build()
            .await;

        let client = reqwest::Client::new();
        client.post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap();

        mock.assert_request_count(1); // should not panic
    }

    #[tokio::test]
    async fn test_request_headers_and_auth() {
        let mock = MockLlmServer::builder()
            .next(Response::text("ok"))
            .build()
            .await;

        let client = reqwest::Client::new();
        client.post(&format!("{}/chat/completions", mock.url()))
            .header("authorization", "Bearer sk-test-key")
            .header("x-custom", "hello")
            .json(&serde_json::json!({"model": "t", "messages": []}))
            .send().await.unwrap();

        let headers = mock.request_headers(0).unwrap();
        assert_eq!(headers["authorization"], "Bearer sk-test-key");
        assert_eq!(headers["x-custom"], "hello");

        mock.assert_auth(0, "Bearer sk-test-key");
    }

    #[tokio::test]
    async fn test_assert_tool_choice() {
        let mock = MockLlmServer::builder()
            .next(Response::text("ok"))
            .build()
            .await;

        let client = reqwest::Client::new();
        client.post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "t",
                "messages": [],
                "tool_choice": "required"
            }))
            .send().await.unwrap();

        mock.assert_tool_choice(0, "required");
    }

    #[tokio::test]
    async fn test_fallback_echo() {
        let mock = MockLlmServer::builder()
            .fallback_echo()
            .build()
            .await;

        let client = reqwest::Client::new();
        let resp: Value = client.post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "echo-test",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .send().await.unwrap().json().await.unwrap();

        let content = resp["choices"][0]["message"]["content"].as_str().unwrap();
        // The echo should contain the original request body
        assert!(content.contains("echo-test"));
        assert!(content.contains("hello"));
    }

    #[test]
    fn test_sync_start() {
        let mock = MockLlmServer::builder()
            .next(Response::text("sync hello"))
            .start();

        // Use a blocking HTTP client
        let client = reqwest::blocking::Client::new();
        let resp: Value = client
            .post(&format!("{}/chat/completions", mock.url()))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();

        assert_eq!(resp["choices"][0]["message"]["content"], "sync hello");
        mock.assert_request_count(1);
        // mock dropped here — server shuts down cleanly
    }
}
