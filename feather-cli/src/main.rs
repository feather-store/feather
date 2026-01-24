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
    },
    Search { 
        db: PathBuf, 
        #[arg(short)] npy: PathBuf, 
        #[arg(long, default_value_t = 5)] k: usize,
        #[arg(long)] type_filter: Option<u8>,
        #[arg(long)] source_filter: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::New { path, dim } => {
            DB::open(&path, dim).ok_or_else(|| anyhow::anyhow!("Failed to create DB"))?;
            println!("Created: {:?}", path);
        }
        Commands::Add { db, id, npy, timestamp, importance, context_type, source, content } => {
            let arr: Array1<f32> = ndarray_npy::read_npy(&npy)?;
            let dim = arr.len();
            let db = DB::open(&db, dim).ok_or_else(|| anyhow::anyhow!("Open failed"))?;
            
            let ts = timestamp.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64
            });

            db.add_with_meta(
                id, arr.as_slice().unwrap(), 
                ts, importance, context_type, 
                source.as_deref(), content.as_deref()
            );
            db.save();
            println!("Added ID {} with metadata", id);
        }
        Commands::Search { db, npy, k, type_filter, source_filter } => {
            let arr: Array1<f32> = ndarray_npy::read_npy(&npy)?;
            let dim = arr.len();
            let db = DB::open(&db, dim).ok_or_else(|| anyhow::anyhow!("Open failed"))?;
            
            let (ids, dists) = if type_filter.is_some() || source_filter.is_some() {
                db.search_with_filter(arr.as_slice().unwrap(), k, type_filter, source_filter.as_deref())
            } else {
                db.search(arr.as_slice().unwrap(), k)
            };

            for (id, dist) in ids.iter().zip(dists.iter()) {
                if *id != 0 || *dist != 0.0 {
                    println!("ID: {}  Score: {:.4}", id, dist);
                }
            }
        }
    }
    Ok(())
}
