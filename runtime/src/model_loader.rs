//! Tracks logical active model for swap / display (weights are loaded by Python / user code, not in Rust).

use std::sync::Arc;

use tokio::sync::RwLock;

#[derive(Clone, Default)]
pub struct ActiveModel {
    pub model_ref: String,
    pub quantization: Option<String>,
}

#[derive(Clone)]
pub struct ModelLoader {
    active: Arc<RwLock<ActiveModel>>,
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelLoader {
    pub fn new() -> Self {
        Self {
            active: Arc::new(RwLock::new(ActiveModel::default())),
        }
    }

    pub async fn swap(&self, model_ref: String, quantization: Option<String>) {
        let mut a = self.active.write().await;
        a.model_ref = model_ref;
        a.quantization = quantization;
    }

    pub async fn get(&self) -> ActiveModel {
        self.active.read().await.clone()
    }
}
