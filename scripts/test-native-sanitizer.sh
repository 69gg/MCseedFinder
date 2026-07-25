#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sanitizer_binary=$(mktemp "${TMPDIR:-/tmp}/mcseed-native-asan.XXXXXX")
trap 'rm -f -- "$sanitizer_binary"' EXIT HUP INT TERM

cd "$project_root"

clang \
    -std=c11 \
    -O1 \
    -g \
    -fno-omit-frame-pointer \
    -fsanitize=address \
    -I native \
    -I vendor/cubiomes \
    native/bridge.c \
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

ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 "$sanitizer_binary"
