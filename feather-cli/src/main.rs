use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs::File;
use feather_cli::DB;
use ndarray::Array1;
use ndarray_npy::read_npy;  // ← FREE FUNCTION

#[derive(Parser)]
#[command(name = "feather")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    New { path: PathBuf, #[arg(long)] dim: usize },
    Add { db: PathBuf, id: u64, #[arg(short)] npy: PathBuf },
    Search { db: PathBuf, #[arg(short)] npy: PathBuf, #[arg(long, default_value_t = 5)] k: usize },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::New { path, dim } => {
            DB::open(&path, dim).ok_or_else(|| anyhow::anyhow!("Failed to create DB"))?;
            println!("Created: {:?}", path);
        }
        Commands::Add { db, id, npy } => {
            let db = DB::open(&db, 0).ok_or_else(|| anyhow::anyhow!("Open failed"))?;
            let file = File::open(&npy)?;
            let arr: Array1<f32> = read_npy(&file)?;  // ← FIXED: free function
            db.add(id, arr.as_slice().unwrap());
            db.save();
            println!("Added ID {}", id);
        }
        Commands::Search { db, npy, k } => {
            let db = DB::open(&db, 0).ok_or_else(|| anyhow::anyhow!("Open failed"))?;
            let file = File::open(&npy)?;
            let arr: Array1<f32> = read_npy(&file)?;  // ← FIXED: free function
            let (ids, dists) = db.search(arr.as_slice().unwrap(), k);
            for (id, dist) in ids.iter().zip(dists.iter()) {
                println!("ID: {}  dist: {:.4}", id, dist);
            }
        }
    }
    Ok(())
}
