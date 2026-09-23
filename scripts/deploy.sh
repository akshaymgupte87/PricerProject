#!/usr/bin/env bash
# Run only on the prepared deployment host, with PRICER_IMAGE set to a digest.
set -euo pipefail

: "${PRICER_IMAGE:?Set PRICER_IMAGE}"
: "${PRICER_DEPLOY_DIR:?Set PRICER_DEPLOY_DIR}"
[[ "$PRICER_DEPLOY_DIR" = /* && "$PRICER_DEPLOY_DIR" != / ]] || {
  echo 'PRICER_DEPLOY_DIR must be an absolute data directory, not /'; exit 1;
}
[[ "$PRICER_IMAGE" =~ ^ghcr.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$ ]] || {
  echo 'PRICER_IMAGE must be a GHCR image pinned by digest'; exit 1;
}
for file in .env artifacts/deep_neural_network.pth products_bm25.sqlite3; do
  test -f "$PRICER_DEPLOY_DIR/$file" || { echo "Missing deployment file: $file"; exit 1; }
done
test -d "$PRICER_DEPLOY_DIR/products_vectorstore" || { echo 'Missing vector store'; exit 1; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
compose=(docker compose --project-name pricer-production
  --env-file "$PRICER_DEPLOY_DIR/.env" -f "$script_dir/../compose.production.yml")
"${compose[@]}" config --quiet
previous_container=$("${compose[@]}" ps --all --quiet api)
previous_image=''
if [[ -n "$previous_container" ]]; then
  previous_image=$(docker inspect --format '{{.Config.Image}}' "$previous_container")
fi

"${compose[@]}" pull api
if ! "${compose[@]}" up --detach --wait --wait-timeout 180 api; then
  "${compose[@]}" logs --tail 100 api || true
  if [[ -n "$previous_image" ]]; then
    echo 'Startup failed; restoring previous image'
    export PRICER_IMAGE="$previous_image"
    "${compose[@]}" up --detach --wait --wait-timeout 180 api
  else
    echo 'First deployment failed; stopping the unhealthy service'
    "${compose[@]}" stop api
  fi
  exit 1
fi
