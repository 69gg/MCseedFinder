# Third-party notices

## Cubiomes

- Project: `xpple/cubiomes`, an actively maintained fork of `Cubitect/cubiomes`
- Source: <https://github.com/xpple/cubiomes>
- Vendored commit: `cf7123b3e4c9c0036e008f925cfdfab991ab7f72`
- Commit date: 2026-07-23
- Local path: `vendor/cubiomes/`
- License: MIT; the original license text is preserved at `vendor/cubiomes/LICENSE`

The vendored revision contains the Minecraft 26.2 biome parameter tree and the `sulfur_caves` biome. MCSeed Finder adds a small C ABI bridge and the missing Nether fossil structure-family placement/biome check in `native/bridge.c`.

The upstream snapshot also carries an MIT-licensed loot helper by ScriptLine under `vendor/cubiomes/loot/`; its license is preserved as `LICENSE_loot_library.h.txt`. That directory embeds MIT-licensed cJSON by Dave Gamble and contributors, with the full notice preserved at the top of `loot/cjson/cJSON.c` and `cJSON.h`. MCSeed Finder does not compile or call the loot helper, but keeps the complete upstream source snapshot and notices intact.

No Mojang game source, class file, mapping, asset, or JAR is copied into the program or used as a build/runtime input.
