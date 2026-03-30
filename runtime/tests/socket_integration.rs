//! Integration tests: daemon socket protocol on a temporary Unix socket.

use std::time::Duration;

use neuronbox_runtime::gpu_manager::GpuManager;
use neuronbox_runtime::model_loader::ModelLoader;
use neuronbox_runtime::server::run_socket_server;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

async fn connect_retry(path: &std::path::Path) -> UnixStream {
    for _ in 0..50 {
        if let Ok(s) = UnixStream::connect(path).await {
            return s;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("failed to connect to {}", path.display());
}

async fn read_one_json_line(stream: UnixStream) -> serde_json::Value {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await.expect("read_line");
    serde_json::from_str(line.trim()).expect("valid JSON")
}

#[tokio::test]
async fn daemon_ping_over_temp_socket() {
    std::env::set_var("NEURONBOX_DISABLE_VRAM_WATCH", "1");
    let dir = tempfile::tempdir().expect("tempdir");
    let sock = dir.path().join("neuron.sock");
    let path = sock.clone();
    let gm = GpuManager::new();
    let ml = ModelLoader::new();
    let server = tokio::spawn(async move {
        let _ = run_socket_server(&path, gm, ml).await;
    });

    let mut stream = connect_retry(&sock).await;
    stream
        .write_all(b"{\"method\":\"ping\"}\n")
        .await
        .expect("write ping");
    let v = read_one_json_line(stream).await;
    assert_eq!(v["response"], "pong");

    server.abort();
}

#[tokio::test]
async fn daemon_version_over_temp_socket() {
    std::env::set_var("NEURONBOX_DISABLE_VRAM_WATCH", "1");
    let dir = tempfile::tempdir().expect("tempdir");
    let sock = dir.path().join("neuron.sock");
    let path = sock.clone();
    let gm = GpuManager::new();
    let ml = ModelLoader::new();
    let server = tokio::spawn(async move {
        let _ = run_socket_server(&path, gm, ml).await;
    });

    let mut stream = connect_retry(&sock).await;
    stream
        .write_all(b"{\"method\":\"version\",\"v\":1}\n")
        .await
        .expect("write version");
    let v = read_one_json_line(stream).await;
    assert_eq!(v["response"], "version_info");
    assert_eq!(v["v"], 1);

    server.abort();
}
