use std::path::PathBuf;

pub fn neuronbox_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".neuronbox")
}

pub fn store_root() -> PathBuf {
    neuronbox_home().join("store")
}
