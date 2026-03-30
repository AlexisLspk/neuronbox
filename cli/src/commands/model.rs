use anyhow::Result;
use clap::Subcommand;

use crate::paths::store_root;
use crate::store::registry::StoreIndex;

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List models registered in index.json.
    List,
}

pub fn model(cmd: ModelCommands) -> Result<()> {
    match cmd {
        ModelCommands::List => {
            let store = store_root();
            let idx = StoreIndex::load(&store)?;
            if idx.models.is_empty() {
                println!("No models in the store. Use `neuron pull org/model`.");
                return Ok(());
            }
            for (id, e) in &idx.models {
                println!("{id} -> {}", e.path.display());
            }
        }
    }
    Ok(())
}
