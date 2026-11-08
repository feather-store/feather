use clap::{Parser, Subcommand};
use std::path::PathBuf;
use feather_db_cli::DB;
use ndarray::Array1;

#[derive(Parser)]
#[command(name = "feather")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    New { path: PathBuf, #[arg(long)] dim: usize },
    Add { 
        db: PathBuf, 
        id: u64, 
        #[arg(short)] npy: PathBuf,
        #[arg(long)] timestamp: Option<i64>,
        #[arg(long, default_value_t = 1.0)] importance: f32,
        #[arg(long, default_value_t = 0)] context_type: u8,
        #[arg(long)] source: Option<String>,
        #[arg(long)] content: Option<String>,
        #[arg(long, default_value = "text")] modality: String,
    },
    Link {
        db: PathBuf,
        from: u64,
        to: u64,
    },
    /// Erase records so nothing of them remains: the file is rewritten, the
    /// WAL rotated away, and records derived from or citing them go too.
    /// Prints the receipt (JSON); exits non-zero if it is not verified.
    Erase {
        db: PathBuf,
        /// namespace_id to erase
        #[arg(long)] namespace: Option<String>,
        /// with --namespace: its dotted children too (a.b -> a.b.c)
        #[arg(long)] prefix: bool,
        /// entity_id to erase (with --namespace: only within it)
        #[arg(long)] entity: Option<String>,
        /// a record id to erase (repeatable)
        #[arg(long = "id")] ids: Vec<u64>,
        /// plan only: print what would be erased, change nothing
        #[arg(long)] dry_run: bool,
    },
    Search { 
        db: PathBuf, 
        #[arg(short)] npy: PathBuf, 
        #[arg(long, default_value_t = 5)] k: usize,
        #[arg(long)] type_filter: Option<u8>,
        #[arg(long)] source_filter: Option<String>,
        #[arg(long, default_value = "text")] modality: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::New { path, dim } => {
            DB::try_open(&path, dim).map_err(|e| anyhow::anyhow!("Failed to create DB: {e}"))?;
            println!("Created: {:?}", path);
        }
        Commands::Add { db, id, npy, timestamp, importance, context_type, source, content, modality } => {
            let arr: Array1<f32> = ndarray_npy::read_npy(&npy)?;
            let dim = arr.len();
            let db = DB::try_open(&db, dim).map_err(|e| anyhow::anyhow!("Open failed: {e}"))?;
            
            let ts = timestamp.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64
            });

            db.try_add_with_meta(
                id, arr.as_slice().unwrap(), 
                ts, importance, context_type, 
                source.as_deref(), content.as_deref(), Some(&modality)
            ).map_err(|e| anyhow::anyhow!("Add failed: {e}"))?;
            db.try_save().map_err(|e| anyhow::anyhow!("Save failed: {e}"))?;
            println!("Added ID {} to modality '{}'", id, modality);
        }
        Commands::Link { db, from, to } => {
            let db = DB::try_open(&db, 0).map_err(|e| anyhow::anyhow!("Open failed: {e}"))?;
            db.try_link(from, to).map_err(|e| anyhow::anyhow!("Link failed: {e}"))?;
            db.try_save().map_err(|e| anyhow::anyhow!("Save failed: {e}"))?;
            println!("Linked {} -> {}", from, to);
        }
        Commands::Erase { db, namespace, prefix, entity, ids, dry_run } => {
            let db = DB::try_open(&db, 0).map_err(|e| anyhow::anyhow!("Open failed: {e}"))?;
            let receipt = db.try_erase(namespace.as_deref(), prefix, entity.as_deref(), &ids, dry_run)
                .map_err(|e| anyhow::anyhow!("Erase failed: {e}"))?;
            println!("{receipt}");
            if !dry_run && !receipt.contains("\"verified\":true") {
                anyhow::bail!("the erasure is not verified; see \"problems\" in the receipt");
            }
        }
        Commands::Search { db, npy, k, type_filter, source_filter, modality } => {
            let arr: Array1<f32> = ndarray_npy::read_npy(&npy)?;
            let dim = arr.len();
            let db = DB::try_open(&db, dim).map_err(|e| anyhow::anyhow!("Open failed: {e}"))?;
            
            let (ids, dists) = if type_filter.is_some() || source_filter.is_some() {
                db.try_search_with_filter(arr.as_slice().unwrap(), k, type_filter, source_filter.as_deref(), Some(&modality))
            } else {
                db.try_search(arr.as_slice().unwrap(), k, Some(&modality))
            }.map_err(|e| anyhow::anyhow!("Search failed: {e}"))?;

            for (id, dist) in ids.iter().zip(dists.iter()) {
                if *id != 0 || *dist != 0.0 {
                    println!("ID: {}  Score: {:.4}", id, dist);
                }
            }
        }
    }
    Ok(())
}
