//! feather-server: the Cloud API's hot routes in Rust, on the same
//! `.feather` files and with the same JSON as feather-api: search, keyword
//! and hybrid search, record reads and writes, and every route an agent
//! session sends (AgentMemory / Pocket / PacketBuilder / FeatherStore over
//! RemoteDB). Everything else stays on the Python server: see README.md.

mod api;
mod ffi;
mod manager;
mod packed;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "feather-server [--host 0.0.0.0] [--port 8000]\n\n\
             Environment: FEATHER_DATA_DIR (/data), FEATHER_DB_DIM (768), FEATHER_API_KEY,\n\
             FEATHER_DEV_MODE=1 (no auth), FEATHER_SERVER_THREADS (cores),\n\
             FEATHER_WRITE_THREADS (8), FEATHER_CHECKPOINT_WAL_MB (64),\n\
             FEATHER_CHECKPOINT_MAX_AGE_S (300), FEATHER_CHECKPOINT_POLL_S (5)."
        );
        return;
    }
    let host = arg(&args, "--host").unwrap_or_else(|| "0.0.0.0".into());
    let port: u16 = arg(&args, "--port").and_then(|p| p.parse().ok()).unwrap_or(8000);

    let api_key = std::env::var("FEATHER_API_KEY").unwrap_or_default();
    let dev = matches!(
        std::env::var("FEATHER_DEV_MODE").unwrap_or_default().to_lowercase().as_str(),
        "1" | "true" | "yes"
    );
    // Fail closed, as feather-api does: an unset key is a startup error,
    // not open access.
    if api_key.is_empty() && !dev {
        eprintln!(
            "FEATHER_API_KEY is not set — refusing to start with authentication disabled.\n  \
             Production:  set FEATHER_API_KEY=<a strong random key>\n  \
             Local dev:   set FEATHER_DEV_MODE=1 to run without auth (never do this on a reachable host)."
        );
        std::process::exit(2);
    }
    if api_key.is_empty() {
        eprintln!("FEATHER_DEV_MODE=1 and no FEATHER_API_KEY: authentication is DISABLED.");
    }

    let data_dir = PathBuf::from(std::env::var("FEATHER_DATA_DIR").unwrap_or_else(|_| "/data".into()));
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln!("cannot create FEATHER_DATA_DIR {}: {e}", data_dir.display());
        std::process::exit(2);
    }
    let app = Arc::new(api::App {
        mgr: manager::Manager::new(data_dir.clone(), env_or("FEATHER_DB_DIM", 768usize)),
        api_key,
        write_slots: tokio::sync::Semaphore::new(env_or("FEATHER_WRITE_THREADS", 8usize).max(1)),
    });

    let mut rt = tokio::runtime::Builder::new_multi_thread();
    rt.enable_all();
    let threads: usize = env_or("FEATHER_SERVER_THREADS", 0usize);
    if threads > 0 {
        rt.worker_threads(threads);
    }
    let rt = rt.build().expect("tokio runtime");

    // Checkpoints off the request path (feather-api's _checkpointer).
    let ckpt_app = app.clone();
    let wal_bytes = (env_or("FEATHER_CHECKPOINT_WAL_MB", 64.0f64) * 1024.0 * 1024.0) as u64;
    let max_age = Duration::from_secs_f64(env_or("FEATHER_CHECKPOINT_MAX_AGE_S", 300.0f64));
    let poll = Duration::from_secs_f64(env_or("FEATHER_CHECKPOINT_POLL_S", 5.0f64).max(0.05));
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let checkpointer = std::thread::Builder::new()
        .name("feather-checkpoint".into())
        .spawn(move || {
            while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = stop_rx.recv_timeout(poll) {
                ckpt_app.mgr.checkpoint_pass(wal_bytes, max_age);
            }
        })
        .expect("checkpoint thread");

    let router = Router::new()
        .route("/health", get(api::health))
        .route("/v1/namespaces", get(api::list_namespaces).post(api::create_namespace))
        .route("/v1/{namespace}/search", post(api::search))
        .route("/v1/{namespace}/keyword_search", post(api::keyword_search))
        .route("/v1/{namespace}/hybrid_search", post(api::hybrid_search))
        .route("/v1/{namespace}/vectors", post(api::add_vector))
        .route("/v1/{namespace}/records/get_many", post(api::get_many))
        .route(
            "/v1/{namespace}/records/{record_id}",
            get(api::get_record).put(api::update_record).delete(api::delete_record),
        )
        .route("/v1/{namespace}/erase", post(api::erase))
        // Agent-memory routes: what AgentMemory / Pocket / PacketBuilder /
        // FeatherStore send over RemoteDB besides search and get.
        .route("/v1/{namespace}/batch", post(api::batch))
        .route("/v1/{namespace}/by_key", get(api::by_key))
        .route("/v1/{namespace}/by_keys", post(api::by_keys))
        .route("/v1/{namespace}/newest", get(api::newest))
        .route("/v1/{namespace}/count_live", get(api::count_live))
        .route("/v1/{namespace}/ids", get(api::ids))
        .route("/v1/{namespace}/metadata_namespaces", get(api::metadata_namespaces))
        .route("/v1/{namespace}/heat_select", post(api::heat_select))
        .route("/v1/{namespace}/records/touch_many", post(api::touch_many))
        .route("/v1/{namespace}/records/{record_id}/touch", post(api::touch))
        .route("/v1/{namespace}/records/{record_id}/link", post(api::link))
        .route("/v1/{namespace}/normalize_vectors", post(api::normalize_vectors))
        .route("/v1/{namespace}/admin/engine", get(api::engine_state))
        .route("/v1/{namespace}/save", post(api::save))
        .fallback(api::not_served)
        .method_not_allowed_fallback(api::method_not_allowed)
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(app.clone());

    let addr: SocketAddr = match format!("{host}:{port}").parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bad --host/--port: {e}");
            std::process::exit(2);
        }
    };
    rt.block_on(async move {
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("cannot listen on {addr}: {e}");
                std::process::exit(2);
            }
        };
        eprintln!("feather-server {} on http://{addr} (data {})", env!("CARGO_PKG_VERSION"), data_dir.display());
        use axum::serve::ListenerExt;
        let listener = listener.tap_io(|tcp| {
            let _ = tcp.set_nodelay(true);
        });
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let ctrl_c = tokio::signal::ctrl_c();
                #[cfg(unix)]
                {
                    let mut term =
                        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM");
                    tokio::select! { _ = ctrl_c => {}, _ = term.recv() => {} }
                }
                #[cfg(not(unix))]
                {
                    let _ = ctrl_c.await;
                }
            })
            .await
            .expect("server");
    });
    // Drain: stop checkpointing, then close every namespace with a final
    // checkpoint (requests still in flight finished with the server).
    let _ = stop_tx.send(());
    let _ = checkpointer.join();
    rt.shutdown_timeout(Duration::from_secs(10));
    app.mgr.close_all();
    eprintln!("feather-server stopped");
}
