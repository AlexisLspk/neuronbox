use std::path::{Path, PathBuf};

/// Directory containing `neuron.yaml`. Rust's `path.parent()` is `Some("")` when the yaml is in
/// the current directory (`neuron.yaml`), which breaks `Command::current_dir` (ENOENT on spawn).
pub fn project_dir_from_yaml(yaml: &Path) -> PathBuf {
    match yaml.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

pub fn neuronbox_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".neuronbox")
}

pub fn store_root() -> PathBuf {
    neuronbox_home().join("store")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn project_dir_yaml_in_cwd_is_dot() {
        let cwd = project_dir_from_yaml(Path::new("neuron.yaml"));
        assert_eq!(cwd, PathBuf::from("."));
    }

    #[test]
    fn project_dir_nested_yaml() {
        let cwd = project_dir_from_yaml(Path::new("proj/neuron.yaml"));
        assert_eq!(cwd, PathBuf::from("proj"));
    }
}
