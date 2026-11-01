#!/bin/sh
# export-and-serve MODEL [server arguments...]: exports MODEL (a Nest name,
# a Hub repository or a directory) for the image's server, once, into
# /cache/exports, then runs the server on it. LINNET_EXPORT_ARGS are the
# export's own options (`--bind S=128`, `--backend python`, ...); a change
# to them or to MODEL exports again.
set -eu

model=${1:?usage: export-and-serve MODEL [server arguments...]}
shift
key=$(printf '%s\n%s\n' "$model" "${LINNET_EXPORT_ARGS:-}" | sha256sum | cut -c1-16)
out=/cache/exports/$LINNET_SERVER/$key

if [ ! -e "$out.done" ]; then
    rm -rf "$out"
    mkdir -p "$out"
    # shellcheck disable=SC2086 # the export's options, split as words
    case $LINNET_SERVER in
    triton) python3 -m linnet.triton export "$model" -o "$out" ${LINNET_EXPORT_ARGS:-} ;;
    vllm) python3 -m linnet.hf export "$model" -o "$out" ${LINNET_EXPORT_ARGS:-} ;;
    esac
    touch "$out.done"
fi

case $LINNET_SERVER in
triton) exec tritonserver --model-repository="$out" "$@" ;;
vllm) exec vllm serve "$out" --served-model-name "$model" "$@" ;;
esac
