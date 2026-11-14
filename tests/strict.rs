//! Strict mode: requests a real OpenAI-compatible server rejects get a 400, not a scripted reply.

use mockllm::{MockLlmServer, Response};
use serde_json::{json, Value};

fn post(mock: &MockLlmServer, body: &Value) -> (u16, Value) {
    let r = reqwest::blocking::Client::new()
        .post(format!("{}/chat/completions", mock.url()))
        .json(body)
        .send()
        .unwrap();
    (r.status().as_u16(), r.json().unwrap())
}

/// A history with one finished tool call whose arguments string is `args`.
fn history(args: &str) -> Value {
    json!({"model": "m", "messages": [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": args}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.rs"},
        {"role": "user", "content": "thanks"}]})
}

#[test]
fn valid_history_is_served() {
    let mock = MockLlmServer::builder().strict().next(Response::text("ok")).start();
    let (status, body) = post(&mock, &history(r#"{"command":"ls"}"#));
    assert_eq!(status, 200);
    assert_eq!(body["choices"][0]["message"]["content"], "ok");
    assert_eq!(mock.rejected_count(), 0);
}

/// The bug this mode exists for: a client that rewrites old tool-call arguments into a
/// non-JSON "stub" passes a lenient mock but gets 400 from vLLM-style servers.
#[test]
fn non_json_arguments_are_rejected_and_keep_the_script() {
    let mock = MockLlmServer::builder().strict().next(Response::text("ok")).start();
    let (status, body) = post(&mock, &history(r#"[argstub v1] {"bytes":120}"#));
    assert_eq!(status, 400);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("messages[1].tool_calls[0].function.arguments is not valid JSON"), "{msg}");
    assert_eq!(mock.rejected_count(), 1);
    assert_eq!(mock.request_count(), 1, "rejected requests are still recorded");
    assert_eq!(mock.remaining_responses(), 1, "a rejected request consumes no scripted response");
}

/// Positive control for the default: the same request is served when strict is off.
#[test]
fn lenient_by_default() {
    let mock = MockLlmServer::builder().next(Response::text("ok")).start();
    let (status, _) = post(&mock, &history(r#"[argstub v1] {"bytes":120}"#));
    assert_eq!(status, 200);
    assert_eq!(mock.rejected_count(), 0);
}

#[test]
fn other_shapes_real_servers_refuse() {
    let mock = MockLlmServer::builder().strict().start();
    let cases = [
        (json!({"model": "m"}), "'messages' must be an array"),
        (json!({"model": "m", "messages": []}), "must not be empty"),
        (json!({"model": "m", "messages": [{"content": "hi"}]}), "missing 'role'"),
        (json!({"model": "m", "messages": [{"role": "robot", "content": "hi"}]}), "unknown role"),
        (json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}), "needs a 'tool_call_id'"),
        (json!({"model": "m", "messages": [{"role": "assistant", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]}]}),
         "must be a string"),
    ];
    for (body, want) in &cases {
        let (status, resp) = post(&mock, body);
        assert_eq!(status, 400, "{body}");
        let msg = resp["error"]["message"].as_str().unwrap();
        assert!(msg.contains(want), "{body}: got {msg:?}, want {want:?}");
    }
    assert_eq!(mock.rejected_count(), cases.len());
}
