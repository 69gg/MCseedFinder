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

## Minecraft village metadata

`native/generated/village_26_2.inc` 是由 `tools/extract_jigsaw.py` 从合法取得的 Minecraft Java Edition 26.2 客户端 JAR 派生的紧凑世界生成元数据。它只记录原版村庄模板池的资源标识、权重、投影、模板边界尺寸及 Jigsaw 连接点属性，不包含方块调色板、方块、实体、战利品、完整结构模板、类文件、反编译源码或映射。生成文件记录了输入 JAR 的 SHA-256，便于复现和审计。

Minecraft 是 Mojang Studios 的商标。本项目并非 Mojang 或 Microsoft 的官方项目。Minecraft JAR、游戏源码和 `sources/` 均不作为构建、测试或运行输入，也不会随 crate 打包。
