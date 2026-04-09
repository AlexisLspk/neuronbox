//! SDK path resolution for auto-hooks.
//!
//! Order of resolution (first match wins):
//! 1. `NEURONBOX_SDK` environment variable (explicit override)
//! 2. Dev layout: `../../..` from exe (for `cargo build` in repo)
//! 3. `~/.neuronbox/sdk` if present (manual install)
//! 4. Bundled SDK extracted to `~/.neuronbox/sdk-bundled/<version>/`
//!
//! The SDK is considered valid if `neuronbox/_hooks.py` exists under the path.

use std::path::{Path, PathBuf};

use crate::paths::neuronbox_home;

/// Marker file that indicates a valid SDK directory.
const SDK_MARKER: &str = "neuronbox/_hooks.py";

/// Check if a path contains a valid SDK (has the marker file).
fn is_valid_sdk(path: &Path) -> bool {
    path.join(SDK_MARKER).exists()
}

/// Check if auto-hooks are explicitly disabled via environment.
pub fn autohook_disabled() -> bool {
    std::env::var_os("NEURONBOX_DISABLE_AUTOHOOK")
        .map(|v| {
            let s = v.to_string_lossy().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
}

/// Resolve the SDK path using the priority order.
/// Returns `None` if no valid SDK is found or auto-hooks are disabled.
pub fn resolve_sdk_path() -> Option<PathBuf> {
    if autohook_disabled() {
        return None;
    }

    // 1. NEURONBOX_SDK environment variable (explicit override)
    if let Some(sdk_env) = std::env::var_os("NEURONBOX_SDK") {
        let path = PathBuf::from(sdk_env);
        if is_valid_sdk(&path) {
            return Some(path);
        }
        // If explicitly set but invalid, warn and continue
        eprintln!(
            "neuron: warning — NEURONBOX_SDK={} does not contain {}",
            path.display(),
            SDK_MARKER
        );
    }

    // 2. Dev layout: exe -> target/debug/neuron -> repo root is ../../..
    if let Some(path) = resolve_dev_layout() {
        return Some(path);
    }

    // 3. ~/.neuronbox/sdk (manual install)
    let user_sdk = neuronbox_home().join("sdk");
    if is_valid_sdk(&user_sdk) {
        return Some(user_sdk);
    }

    // 4. Bundled SDK extracted to ~/.neuronbox/sdk-bundled/<version>/
    if let Some(path) = resolve_bundled_sdk() {
        return Some(path);
    }

    None
}

/// Try to find SDK in dev layout (relative to binary).
fn resolve_dev_layout() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    // target/debug/neuron -> repo root is ../../..
    let repo_root = exe.parent()?.parent()?.parent()?;
    let sdk_path = repo_root.join("sdk");
    if is_valid_sdk(&sdk_path) {
        return Some(sdk_path);
    }
    None
}

/// Try to find bundled SDK extracted to ~/.neuronbox/sdk-bundled/<version>/.
fn resolve_bundled_sdk() -> Option<PathBuf> {
    let bundled_root = neuronbox_home().join("sdk-bundled");
    let version = env!("CARGO_PKG_VERSION");
    let versioned_path = bundled_root.join(version);
    if is_valid_sdk(&versioned_path) {
        return Some(versioned_path);
    }
    None
}

/// Extract bundled SDK to disk if not already present.
/// Called by `neuron run` when no other SDK path is found.
/// Returns the path to the extracted SDK, or None if extraction fails.
#[cfg(feature = "bundled-sdk")]
pub fn ensure_bundled_sdk() -> Option<PathBuf> {
    use include_dir::{include_dir, Dir};

    static SDK_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/sdk");

    let bundled_root = neuronbox_home().join("sdk-bundled");
    let version = env!("CARGO_PKG_VERSION");
    let versioned_path = bundled_root.join(version);

    // Already extracted?
    if is_valid_sdk(&versioned_path) {
        return Some(versioned_path);
    }

    // Extract
    if let Err(e) = std::fs::create_dir_all(&versioned_path) {
        eprintln!(
            "neuron: warning — failed to create SDK directory {}: {}",
            versioned_path.display(),
            e
        );
        return None;
    }

    if let Err(e) = SDK_DIR.extract(&versioned_path) {
        eprintln!(
            "neuron: warning — failed to extract bundled SDK to {}: {}",
            versioned_path.display(),
            e
        );
        return None;
    }

    if is_valid_sdk(&versioned_path) {
        Some(versioned_path)
    } else {
        None
    }
}

/// Stub for when bundled-sdk feature is not enabled.
#[cfg(not(feature = "bundled-sdk"))]
pub fn ensure_bundled_sdk() -> Option<PathBuf> {
    None
}

/// Get the SDK path, extracting bundled SDK if necessary.
/// This is the main entry point for `neuron run` and `neuron serve`.
pub fn get_sdk_path() -> Option<PathBuf> {
    // First try normal resolution
    if let Some(path) = resolve_sdk_path() {
        return Some(path);
    }

    // If no SDK found, try to extract bundled SDK
    ensure_bundled_sdk()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_is_valid_sdk() {
        let tmp = TempDir::new().unwrap();
        let sdk_path = tmp.path().to_path_buf();

        // Not valid without marker
        assert!(!is_valid_sdk(&sdk_path));

        // Create marker
        fs::create_dir_all(sdk_path.join("neuronbox")).unwrap();
        fs::write(sdk_path.join("neuronbox/_hooks.py"), "# hooks").unwrap();

        assert!(is_valid_sdk(&sdk_path));
    }

    #[test]
    fn test_autohook_disabled() {
        // Save original
        let orig = std::env::var_os("NEURONBOX_DISABLE_AUTOHOOK");

        std::env::remove_var("NEURONBOX_DISABLE_AUTOHOOK");
        assert!(!autohook_disabled());

        std::env::set_var("NEURONBOX_DISABLE_AUTOHOOK", "1");
        assert!(autohook_disabled());

        std::env::set_var("NEURONBOX_DISABLE_AUTOHOOK", "true");
        assert!(autohook_disabled());

        std::env::set_var("NEURONBOX_DISABLE_AUTOHOOK", "yes");
        assert!(autohook_disabled());

        std::env::set_var("NEURONBOX_DISABLE_AUTOHOOK", "0");
        assert!(!autohook_disabled());

        // Restore
        match orig {
            Some(v) => std::env::set_var("NEURONBOX_DISABLE_AUTOHOOK", v),
            None => std::env::remove_var("NEURONBOX_DISABLE_AUTOHOOK"),
        }
    }

    #[test]
    fn test_env_override() {
        let tmp = TempDir::new().unwrap();
        let sdk_path = tmp.path().to_path_buf();

        // Create valid SDK
        fs::create_dir_all(sdk_path.join("neuronbox")).unwrap();
        fs::write(sdk_path.join("neuronbox/_hooks.py"), "# hooks").unwrap();

        // Save original
        let orig = std::env::var_os("NEURONBOX_SDK");

        std::env::set_var("NEURONBOX_SDK", &sdk_path);
        let resolved = resolve_sdk_path();
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap(), sdk_path);

        // Restore
        match orig {
            Some(v) => std::env::set_var("NEURONBOX_SDK", v),
            None => std::env::remove_var("NEURONBOX_SDK"),
        }
    }
}
