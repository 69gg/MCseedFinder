#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sanitizer_binary=$(mktemp "${TMPDIR:-/tmp}/mcseed-native-asan.XXXXXX")
trap 'rm -f -- "$sanitizer_binary"' EXIT HUP INT TERM

detect_leaks=${MCSEED_ASAN_DETECT_LEAKS:-1}
case "$detect_leaks" in
    0|1) ;;
    *)
        echo "MCSEED_ASAN_DETECT_LEAKS 只能是 0 或 1" >&2
        exit 2
        ;;
esac

cd "$project_root"

# Cubiomes intentionally stores ABI-compatible RNG callbacks behind generic function pointers.
clang \
    -std=c11 \
    -O1 \
    -g \
    -Wall \
    -Wextra \
    -Werror \
    -Wno-unused-parameter \
    -Wno-unused-function \
    -fno-omit-frame-pointer \
    -fsanitize=address,undefined \
    -fno-sanitize=function \
    -I native \
    -I vendor/cubiomes \
    native/bridge.c \
    native/gpu/reference.c \
    native/jigsaw.c \
    vendor/cubiomes/biomenoise.c \
    vendor/cubiomes/biomes.c \
    vendor/cubiomes/finders.c \
    vendor/cubiomes/generator.c \
    vendor/cubiomes/layers.c \
    vendor/cubiomes/noise.c \
    vendor/cubiomes/quadbase.c \
    vendor/cubiomes/terrainnoise.c \
    vendor/cubiomes/util.c \
    vendor/cubiomes/xradv.c \
    vendor/cubiomes/features/stronghold.c \
    tests/native_smoke.c \
    -lm \
    -o "$sanitizer_binary"

ASAN_OPTIONS="detect_leaks=${detect_leaks}:abort_on_error=1" \
UBSAN_OPTIONS=halt_on_error=1 \
    "$sanitizer_binary"
