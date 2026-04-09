//! Build script for neuronbox-cli.
//!
//! When the `bundled-sdk` feature is enabled, this validates that the bundled SDK
//! directory exists inside the CLI crate package.

fn main() {
    // Re-run if bundled SDK files change
    println!("cargo:rerun-if-changed=sdk/neuronbox/_hooks.py");
    println!("cargo:rerun-if-changed=sdk/neuronbox/__init__.py");
    println!("cargo:rerun-if-changed=sdk/neuronbox/client.py");
    println!("cargo:rerun-if-changed=sdk/neuronbox/sitecustomize.py");

    #[cfg(feature = "bundled-sdk")]
    {
        let sdk_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sdk");
        let hooks_path = sdk_path.join("neuronbox/_hooks.py");
        assert!(
            hooks_path.exists(),
            "bundled-sdk feature requires SDK at {:?}, but {} not found",
            sdk_path.display(),
            hooks_path.display()
        );
    }
}
