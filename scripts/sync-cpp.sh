#!/usr/bin/env bash
# Mirror the canonical engine into the Rust CLI's vendored copy.
#
# feather-cli/cpp/ exists because `cargo package` refuses to follow paths outside
# the package root, so build.rs cannot compile against ../../include directly and
# `cargo install feather-db-cli` would break for every downstream user if it did.
# The copy is therefore GENERATED, never hand-edited.
#
# Left to drift it becomes dangerous rather than merely stale: at v0.17.0 the
# vendored engine was 764 lines behind and contained none of the WAL recovery,
# fsync, checksum, failed-open or salience fixes — so a published CLI would still
# lose a bulk import on crash, under a version whose notes said otherwise.
#
# Run this before tagging a release. CI runs it and fails on any diff.
set -euo pipefail
cd "$(dirname "$0")/.."
FILES=(
  include/bruteforce.h include/feather.h include/space_l2.h
  include/visited_list_pool.h include/stop_condition.h include/space_ip.h
  include/hnswalg.h include/hnswlib.h include/metadata.h include/filter.h
  include/scoring.h
  src/filter.cpp src/scoring.cpp src/metadata.cpp src/feather_core.cpp
)
for f in "${FILES[@]}"; do
  mkdir -p "feather-cli/cpp/$(dirname "$f")"
  cp "$f" "feather-cli/cpp/$f"
done
echo "synced ${#FILES[@]} files into feather-cli/cpp/"
