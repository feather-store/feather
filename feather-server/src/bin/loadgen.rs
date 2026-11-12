//! feather-loadgen: an open-loop HTTP load generator (fixed arrival rate).
//!
//! Requests are scheduled at exact times (start + i / rate) by a dedicated
//! timer thread, whatever the server is doing, and handed to `--conns`
//! keep-alive connections. Latency is measured from the SCHEDULED time, so
//! when the server falls behind, the queueing it causes is counted instead of
//! silently lowering the offered rate (coordinated omission). `service_us` is
//! the same requests measured from when they were actually sent.
//!
//!   feather-loadgen --url http://127.0.0.1:8000/v1/bench/search \
//!       --bodies bodies.jsonl --rate 2000 --duration 20 [--conns 64] \
//!       [--api-key K] [--method POST] [--timeout 10]
//!
//! `--bodies`: one request body per line, sent in turn. Prints one JSON
//! object with the results.
//!
//! `--requests`: instead, one `{"method", "path", "body"}` per line (body a
//! JSON string or null), sent in turn to the host of `--url`: a recorded mix
//! of routes, such as an agent turn. `--ok-statuses 200,201,404` counts those
//! statuses as served (default: any 2xx); a mix whose reads of not-yet-written
//! keys answer 404 needs it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper_util::rt::TokioIo;

struct Args {
    url: String,
    bodies: String,
    rate: f64,
    duration: f64,
    conns: usize,
    api_key: String,
    method: String,
    timeout: f64,
    threads: usize,
    requests: String,
    ok_statuses: Vec<u16>,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let get = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1).cloned());
    let need = |f: &str| {
        get(f).unwrap_or_else(|| {
            eprintln!("missing {f} (see the header of src/bin/loadgen.rs)");
            std::process::exit(2)
        })
    };
    Args {
        url: need("--url"),
        bodies: get("--bodies").unwrap_or_default(),
        rate: need("--rate").parse().expect("--rate"),
        duration: need("--duration").parse().expect("--duration"),
        conns: get("--conns").map(|v| v.parse().expect("--conns")).unwrap_or(64),
        api_key: get("--api-key").unwrap_or_default(),
        method: get("--method").unwrap_or_else(|| "POST".into()),
        timeout: get("--timeout").map(|v| v.parse().expect("--timeout")).unwrap_or(10.0),
        threads: get("--threads").map(|v| v.parse().expect("--threads")).unwrap_or(2),
        requests: get("--requests").unwrap_or_default(),
        ok_statuses: get("--ok-statuses")
            .map(|v| v.split(',').map(|s| s.trim().parse().expect("--ok-statuses")).collect())
            .unwrap_or_default(),
    }
}

/// One finished request: (scheduled → done, sent → done) in µs, and status.
struct Sample {
    total_us: u64,
    service_us: u64,
    status: u16,
    /// Scheduled time, µs after the first slot (for LOADGEN_TRACE).
    at_us: u64,
}

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn summary(mut v: Vec<u64>) -> serde_json::Value {
    v.sort_unstable();
    let mean = if v.is_empty() { 0.0 } else { v.iter().sum::<u64>() as f64 / v.len() as f64 };
    serde_json::json!({
        "p50": pct(&v, 50.0), "p90": pct(&v, 90.0), "p99": pct(&v, 99.0),
        "p999": pct(&v, 99.9), "max": v.last().copied().unwrap_or(0), "mean": mean.round(),
    })
}

fn main() {
    let args = parse_args();
    let uri: hyper::Uri = args.url.parse().expect("--url");
    let host = uri.host().expect("url host").to_string();
    let port = uri.port_u16().unwrap_or(80);
    let path = uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".into());
    let bodies: Vec<Bytes> = if args.bodies.is_empty() {
        vec![Bytes::new()]
    } else {
        std::fs::read_to_string(&args.bodies)
            .expect("--bodies")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| Bytes::from(l.to_string()))
            .collect()
    };
    let method: hyper::Method = args.method.parse().expect("--method");
    // What each request sends: one route with varying bodies, or a recorded mix.
    let reqs: Vec<(hyper::Method, String, Bytes)> = if args.requests.is_empty() {
        bodies.into_iter().map(|b| (method.clone(), path.clone(), b)).collect()
    } else {
        std::fs::read_to_string(&args.requests)
            .expect("--requests")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("--requests: a JSON object per line");
                let m = v["method"].as_str().expect("method").parse().expect("method");
                let p = v["path"].as_str().expect("path").to_string();
                let b = v["body"].as_str().map(|b| Bytes::from(b.to_string())).unwrap_or_default();
                (m, p, b)
            })
            .collect()
    };
    let reqs = Arc::new(reqs);
    let total = (args.rate * args.duration).round() as usize;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads.max(1))
        .enable_all()
        .build()
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Instant>();
    let rx = Arc::new(tokio::sync::Mutex::new(rx));
    let next_body = Arc::new(AtomicUsize::new(0));
    let timeout = Duration::from_secs_f64(args.timeout);

    let start = Instant::now() + Duration::from_millis(200); // connections first
    let samples = rt.block_on(async move {
        let mut workers = vec![];
        for _ in 0..args.conns {
            let (rx, reqs, next_body, host, key) =
                (rx.clone(), reqs.clone(), next_body.clone(), host.clone(), args.api_key.clone());
            workers.push(tokio::spawn(async move {
                let mut out: Vec<Sample> = vec![];
                let mut sender: Option<hyper::client::conn::http1::SendRequest<Full<Bytes>>> = None;
                loop {
                    let Some(scheduled) = rx.lock().await.recv().await else { break };
                    if sender.as_ref().is_none_or(|s| s.is_closed()) {
                        let stream = match tokio::net::TcpStream::connect((host.as_str(), port)).await {
                            Ok(s) => s,
                            Err(_) => {
                                out.push(Sample {
                                    total_us: u64::MAX,
                                    service_us: u64::MAX,
                                    status: 0,
                                    at_us: scheduled.saturating_duration_since(start).as_micros() as u64,
                                });
                                continue;
                            }
                        };
                        let _ = stream.set_nodelay(true);
                        let (s, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream)).await.unwrap();
                        tokio::spawn(conn);
                        sender = Some(s);
                    }
                    let s = sender.as_mut().unwrap();
                    let (method, path, body) = &reqs[next_body.fetch_add(1, Ordering::Relaxed) % reqs.len()];
                    let body = body.clone();
                    let mut req = Request::builder()
                        .method(method.clone())
                        .uri(path.as_str())
                        .header("host", host.as_str())
                        .header("content-type", "application/json");
                    if !key.is_empty() {
                        req = req.header("x-api-key", key.as_str());
                    }
                    let req = req.body(Full::new(body)).unwrap();
                    let sent = Instant::now();
                    let status = match tokio::time::timeout(timeout, async {
                        let resp = s.send_request(req).await.ok()?;
                        let st = resp.status().as_u16();
                        resp.into_body().collect().await.ok()?;
                        Some(st)
                    })
                    .await
                    {
                        Ok(Some(st)) => st,
                        _ => {
                            sender = None; // a timed-out or broken connection is not reused
                            0
                        }
                    };
                    let done = Instant::now();
                    out.push(Sample {
                        total_us: done.duration_since(scheduled).as_micros() as u64,
                        service_us: done.duration_since(sent).as_micros() as u64,
                        status,
                        at_us: scheduled.saturating_duration_since(start).as_micros() as u64,
                    });
                }
                out
            }));
        }

        // The schedule runs on its own OS thread: sleep to ~100 µs before
        // each slot, then spin, so the offered rate does not depend on the
        // async runtime's timer granularity (1 ms) or on how busy it is.
        let rate = args.rate;
        let scheduler = std::thread::spawn(move || {
            let gap = Duration::from_secs_f64(1.0 / rate);
            for i in 0..total {
                let at = start + gap.mul_f64(i as f64);
                loop {
                    let now = Instant::now();
                    if now >= at {
                        break;
                    }
                    let left = at - now;
                    if left > Duration::from_micros(200) {
                        std::thread::sleep(left - Duration::from_micros(100));
                    } else {
                        std::hint::spin_loop();
                    }
                }
                if tx.send(at).is_err() {
                    break;
                }
            }
            // tx dropped here: workers finish the queue, then exit.
        });

        let mut all = vec![];
        for w in workers {
            all.extend(w.await.unwrap());
        }
        scheduler.join().unwrap();
        all
    });
    let wall = Instant::now().duration_since(start).as_secs_f64();
    // LOADGEN_TRACE=<file>: every request as "scheduled_us,service_us,total_us,status".
    if let Ok(path) = std::env::var("LOADGEN_TRACE") {
        let mut rows: Vec<&Sample> = samples.iter().collect();
        rows.sort_by_key(|s| s.at_us);
        let text: String =
            rows.iter().map(|s| format!("{},{},{},{}\n", s.at_us, s.service_us, s.total_us, s.status)).collect();
        std::fs::write(&path, text).expect("LOADGEN_TRACE");
    }

    let mut statuses: BTreeMap<String, usize> = BTreeMap::new();
    let (mut ok_total, mut ok_service) = (vec![], vec![]);
    for s in &samples {
        *statuses.entry(s.status.to_string()).or_default() += 1;
        let served = if args.ok_statuses.is_empty() {
            (200..300).contains(&s.status)
        } else {
            args.ok_statuses.contains(&s.status)
        };
        if served {
            ok_total.push(s.total_us);
            ok_service.push(s.service_us);
        }
    }
    let ok = ok_total.len();
    let out = serde_json::json!({
        "url": args.url,
        "target_rate": args.rate,
        "duration_s": args.duration,
        "scheduled": total,
        "ok": ok,
        "errors": total - ok,
        "statuses": statuses,
        "wall_s": wall,
        "achieved_rate": ok as f64 / wall,
        "latency_us": summary(ok_total),
        "service_us": summary(ok_service),
        "conns": args.conns,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
