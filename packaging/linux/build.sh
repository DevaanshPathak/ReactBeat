#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="reactbeat-linux-builder"
OUT_DIR="$ROOT/dist/linux"

docker build --target build -t "$IMAGE" -f "$ROOT/packaging/linux/Dockerfile" "$ROOT"

container_id="$(docker create "$IMAGE")"
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT

mkdir -p "$OUT_DIR"
docker cp "$container_id:/app/dist/reactbeat" "$OUT_DIR/reactbeat"
chmod +x "$OUT_DIR/reactbeat"

docker build --target validate -t reactbeat-linux-validate -f "$ROOT/packaging/linux/Dockerfile" "$ROOT"

echo "Built $OUT_DIR/reactbeat"
