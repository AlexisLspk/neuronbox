//! In-memory session registry (soft VRAM accounting per registered PID).

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::protocol::SessionInfo;

#[derive(Default)]
pub struct GpuManagerState {
    sessions: HashMap<u32, SessionInfo>,
}

#[derive(Clone)]
pub struct GpuManager {
    inner: Arc<RwLock<GpuManagerState>>,
}

impl Default for GpuManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(GpuManagerState::default())),
        }
    }

    pub async fn register(&self, info: SessionInfo) {
        let mut g = self.inner.write().await;
        g.sessions.insert(info.pid, info);
    }

    pub async fn unregister(&self, pid: u32) -> bool {
        let mut g = self.inner.write().await;
        g.sessions.remove(&pid).is_some()
    }

    pub async fn list(&self) -> Vec<SessionInfo> {
        let g = self.inner.read().await;
        g.sessions.values().cloned().collect()
    }

    pub async fn total_estimated_vram_mb(&self) -> u64 {
        let g = self.inner.read().await;
        g.sessions.values().map(|s| s.estimated_vram_mb).sum()
    }
}
