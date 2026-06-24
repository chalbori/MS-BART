#!/usr/bin/env bash
# Download the MS-BART + MIST weights archive from Google Drive and place it at
# deploy/weights/ so that:
#     deploy/weights/msbart/        (MS-BART model + tokenizer, ~399 MB)
#     deploy/weights/mist.ckpt      (MIST fingerprint model, ~58 MB)
#
# The archive on Drive must be a single .zip or .tar.gz that contains those two
# artifacts somewhere inside it (any nesting is fine — we locate them).
#
# Make the archive locally with:
#     cd deploy && zip -r msbart-weights.zip weights/
# then upload msbart-weights.zip to Google Drive and share it ("anyone with link").
#
# Usage:
#     bash fetch_weights.sh "<google-drive-link-or-id>"   [DEST_DIR]
#     WEIGHTS_URL=<link> bash fetch_weights.sh
set -euo pipefail

# Fill this in once the weights archive is uploaded to Google Drive, so the
# script can be run with no arguments. A link/id passed on the command line or
# via $WEIGHTS_URL still overrides it.
DEFAULT_WEIGHTS_URL=""   # TODO: paste the Google Drive share link or file id

URL="${1:-${WEIGHTS_URL:-$DEFAULT_WEIGHTS_URL}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${2:-$SCRIPT_DIR/../weights}"

if [ -z "$URL" ]; then
    echo "ERROR: no Google Drive link/id given." >&2
    echo "Usage: bash fetch_weights.sh \"<drive-link-or-id>\" [DEST_DIR]" >&2
    exit 1
fi

if ! command -v gdown >/dev/null 2>&1; then
    echo "ERROR: gdown not installed. Run: pip install gdown" >&2
    exit 1
fi

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch_weights] downloading from Google Drive ..."
# --fuzzy lets gdown accept a full share URL; it handles the large-file confirm token.
gdown --fuzzy "$URL" -O "$TMP/archive"

echo "[fetch_weights] extracting ..."
if head -c 4 "$TMP/archive" | grep -q "PK"; then
    unzip -q "$TMP/archive" -d "$TMP/unpacked"
else
    mkdir -p "$TMP/unpacked"
    tar -xzf "$TMP/archive" -C "$TMP/unpacked"
fi

# Locate the two artifacts wherever they landed and move them into place.
CKPT="$(find "$TMP/unpacked" -name 'mist.ckpt' -type f | head -n1)"
MSBART_DIR="$(dirname "$(find "$TMP/unpacked" -name 'config.json' -path '*msbart*' -type f | head -n1)")"

if [ -z "$CKPT" ] || [ -z "$MSBART_DIR" ] || [ "$MSBART_DIR" = "." ]; then
    echo "ERROR: archive did not contain mist.ckpt and an msbart/ model dir." >&2
    echo "       Contents:" >&2
    find "$TMP/unpacked" -maxdepth 3 >&2
    exit 1
fi

rm -rf "$DEST/msbart"
cp -r "$MSBART_DIR" "$DEST/msbart"
cp "$CKPT" "$DEST/mist.ckpt"

echo "[fetch_weights] done:"
echo "  $DEST/msbart      ($(du -sh "$DEST/msbart" | cut -f1))"
echo "  $DEST/mist.ckpt   ($(du -sh "$DEST/mist.ckpt" | cut -f1))"
