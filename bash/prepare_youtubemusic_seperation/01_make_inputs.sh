#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "${REPO_ROOT}"

mkdir -p "${RUN_DIR}"
tmp_inputs="${INPUTS_FILE}.tmp"
: > "${tmp_inputs}"

if [[ -n "${DATES}" ]]; then
  for date in ${DATES}; do
    echo "listing ${SOURCE_PREFIX}/${date}" >&2
    "${RCLONE}" --config "${RCLONE_CONFIG}" lsf -R --files-only "${SOURCE_PREFIX}/${date}" \
      | awk -v prefix="${SOURCE_PREFIX}" -v date="${date}" '
          BEGIN { IGNORECASE=1 }
          /\.(aac|aif|aiff|flac|m4a|mp3|mp4|ogg|opus|webm|wav)$/ {
            print prefix "/" date "/" $0
          }
        ' >> "${tmp_inputs}"
  done
else
  echo "listing ${SOURCE_PREFIX}" >&2
  "${RCLONE}" --config "${RCLONE_CONFIG}" lsf -R --files-only "${SOURCE_PREFIX}" \
    | awk -v prefix="${SOURCE_PREFIX}" '
        BEGIN { IGNORECASE=1 }
        /\.(aac|aif|aiff|flac|m4a|mp3|mp4|ogg|opus|webm|wav)$/ {
          print prefix "/" $0
        }
      ' >> "${tmp_inputs}"
fi

sort -u "${tmp_inputs}" > "${INPUTS_FILE}"
rm -f "${tmp_inputs}"

echo "inputs_file=${INPUTS_FILE}"
wc -l "${INPUTS_FILE}"
awk '
  BEGIN { IGNORECASE=1 }
  {
    ext=tolower($0)
    sub(/^.*\./, ".", ext)
    count[ext]++
  }
  END {
    for (ext in count) {
      print ext, count[ext]
    }
  }
' "${INPUTS_FILE}" | sort
