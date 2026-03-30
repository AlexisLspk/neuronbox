//! Local `index.json` for resolved models in the store.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StoreIndex {
    #[serde(default)]
    pub models: BTreeMap<String, IndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub path: PathBuf,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub source: String,
}

impl StoreIndex {
    pub fn path(store_root: &Path) -> PathBuf {
        store_root.join("index.json")
    }

    pub fn load(store_root: &Path) -> Result<Self> {
        let p = Self::path(store_root);
        if !p.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&p).with_context(|| format!("read {:?}", p))?;
        let idx: StoreIndex = serde_json::from_str(&raw).unwrap_or_default();
        Ok(idx)
    }

    pub fn save(&self, store_root: &Path) -> Result<()> {
        let p = Self::path(store_root);
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let raw = serde_json::to_string_pretty(self)?;
        std::fs::write(&p, raw).with_context(|| format!("write {:?}", p))?;
        Ok(())
    }
}
