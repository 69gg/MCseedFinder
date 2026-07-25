# MCSeed Finder

面向 Minecraft Java Edition **26.2** 的纯命令行种子筛选器。它可以按出生生物群系、出生点周边生物群系、主世界结构、出生点投影到下界后的附近结构，以及末地结构组合筛选种子。

世界生成核心随项目一起放在 `vendor/cubiomes/`，构建时编译为本地静态代码。程序的构建、运行和测试都不会读取 `sources/`、Minecraft JAR、游戏安装目录或其他被 Git 忽略的输入。

## 构建

需要 Rust 工具链和支持 C11 的 C 编译器：

```bash
cargo build --release
./target/release/mcseed-finder --help
```

第一次构建需要从 crates.io 获取 Rust 通用依赖。Minecraft 世界生成实现已经纳入仓库，不会在构建或运行时下载游戏文件。

## 常用命令

列出程序实际接受的名称：

```bash
mcseed-finder list biomes
mcseed-finder list structures
mcseed-finder list structures --json
```

检查单个种子：

```bash
mcseed-finder check 0 --spawn-biome plains,forest
```

搜索出生在平原或森林、出生点 1024 格内有村庄，并且进入下界后 512 格内有堡垒或堡垒遗迹的种子：

```bash
mcseed-finder find \
  --start 0 \
  --end 1000000 \
  --count 5 \
  --spawn-biome plains,forest \
  --structure-near village:1024 \
  --structure-near fortress,bastion_remnant:512 \
  --progress
```

随机搜索完整 64 位种子空间，并持续到找到 10 个结果：

```bash
mcseed-finder find \
  --random \
  --count 10 \
  --spawn-biome plains,forest \
  --structure-near village:1024 \
  --progress
```

程序默认使用系统报告的全部可用逻辑 CPU。也可以显式指定线程数，并用随机化键复现一次搜索：

```bash
mcseed-finder find \
  --config examples/random_search.json \
  --threads 12 \
  --progress
```

搜索周边生物群系：

```bash
# 主世界出生点 2000 格内，在 Y=64 处存在蘑菇岛
mcseed-finder find --biome-near overworld:mushroom_fields:2000:64

# 下界出生投影点 512 格内，在 Y=32..96 存在诡异森林或绯红森林
mcseed-finder find --biome-near nether:warped_forest,crimson_forest:512:32..96

# 26.2 新增硫磺洞穴
mcseed-finder check 0 --biome-near overworld:sulfur_caves:1024:-64..64
```

命令行快捷条件的规则：

- `--spawn-biome BIOME[,BIOME]`：逗号内为“任一”。
- `--biome-near DIMENSION:BIOME[,BIOME]:RADIUS[:Y|Y_MIN..Y_MAX]`。
- `--structure-near STRUCTURE[,STRUCTURE]:RADIUS`：结构自身决定维度；主世界默认以出生点为中心，下界默认以 `floor(spawn / 8)` 为中心，末地默认以 `(0, 0)` 为中心。
- 同一种快捷参数可以重复，重复的条件之间为“并且”。
- 顺序模式使用半开区间 `[start, end)`；每批结果会按遍历位置合并，因此不同线程数会返回相同的最小匹配种子。
- `find --random` 使用完整有符号 64 位种子空间的伪随机全排列，不会重复抽取种子。省略 `--random-seed` 时会生成随机化键并立即写到 stderr；使用相同键和条件可以复现相同结果顺序。
- 命令行显式传入 `--random` 时会覆盖配置文件中的 `start`/`end`，便于复用只为顺序搜索编写的条件文件；JSON 自身若设置 `"random": true`，则不能再包含 `start`/`end`。
- `--count N` 指定目标结果数。随机模式省略 `--max-attempts` 时会持续搜索到找满，或在理论上遍历完全部 2^64 个种子；可以随时按 Ctrl+C 停止。
- `--threads N` 控制并行度，默认使用 `available_parallelism` 返回的可用逻辑 CPU 数。工作线程和各自的世界生成上下文会在整次搜索中复用。
- `--batch-size` 控制任务派发粒度，默认 4096。通常无需调整；更小的值会缩短停止响应批次，更大的值适合单个条件非常便宜的搜索。
- `--format jsonl` 适合交给其他程序继续处理。匹配失败时 `check` 和没有结果的 `find` 返回退出码 1，参数或运行错误返回 2。

## JSON 配置

复杂条件使用 JSON。命令行条件会追加到配置中的 `conditions`：

```bash
mcseed-finder find --config examples/village_and_nether.json
mcseed-finder check 0 --config examples/village_and_nether.json --format jsonl
mcseed-finder find --config examples/random_search.json
```

基本格式：

```json
{
  "version": "26.2",
  "search": {
    "start": 0,
    "end": 1000000,
    "results": 5,
    "threads": 8,
    "batch_size": 4096
  },
  "conditions": [
    {
      "type": "spawn_biome",
      "any_of": ["plains", "forest"]
    },
    {
      "type": "structure_near",
      "any_of": ["village"],
      "radius": 1024,
      "min_count": 1
    },
    {
      "type": "biome_near",
      "dimension": "overworld",
      "any_of": ["mushroom_fields"],
      "anchor": "spawn",
      "radius": 2000,
      "y": 64
    }
  ]
}
```

JSON 中也可以启用随机模式。随机模式不能同时设置 `start`/`end`：

```json
{
  "search": {
    "random": true,
    "random_seed": 20260725,
    "max_attempts": 100000000,
    "results": 5,
    "threads": 12,
    "batch_size": 4096
  }
}
```

支持的条件：

- `spawn_biome`：字段 `any_of`。
- `biome_near`：字段 `dimension`、`any_of`、`radius`，可选 `anchor`、`y` 或 `y_min`/`y_max`、`min_count`、`max_count`。
- `structure_near`：字段 `any_of`、`radius`，可选 `dimension`、`anchor`、`min_count`、`max_count`。
- `all` / `any`：字段 `conditions`，可任意递归组合。
- `not`：字段 `condition`。

`anchor` 可以是 `origin`、`spawn`、`nether_spawn`，也可以是自定义坐标，例如 `{"x": 1200, "z": -300}`。用 `min_count: 0, max_count: 0` 可以表达“范围内不能有该目标”。配置采用严格字段校验，拼错的字段会直接报错。

## 支持范围和精度

`list structures` 列出 26.2 的全部内置结构族：村庄、沙漠神殿、雪屋、丛林神庙、沼泽小屋、掠夺者前哨站、远古城市、海底神殿、林地府邸、埋藏的宝藏、废弃矿井、废弃传送门、沉船、海底废墟、下界要塞、堡垒遗迹、下界化石、末地城、要塞、古迹废墟和试炼密室；废弃传送门按主世界和下界分别提供名称。

- 生物群系范围按 Minecraft 的 4×4×4 噪声生物群系单元扫描；圆形半径按水平方块距离判断。跨 Y 范围会检查所有相交的生物群系单元。
- 出生位置使用 26.2 气候适应度搜索和近似地表检查；出生生物群系会在求得的近似地表 Y 上以方块分辨率读取，避免把地下洞穴群系当成地表出生群系。由于不生成完整方块区块，极少数种子的最终落脚方块仍可能与游戏有少量偏差。
- 大部分结构会执行随机放置、生物群系和已实现的地形可用性检查。沙漠神殿、丛林神庙、林地府邸的地表高度是近似检查；废弃传送门和下界化石还可能因完整区块内部地形而生成失败。`list structures` 会逐项显示精度等级。
- 这里筛选的是默认世界类型；不支持大型生物群系、数据包改写的世界生成或自定义维度。

如果要在游戏中长期使用找到的种子，建议先用 `check` 保存见证坐标，再在原版 26.2 中做一次最终确认。

## 测试

```bash
cargo test --offline
cargo clippy --all-targets --offline -- -D warnings
sh scripts/test-native-sanitizer.sh
```

最后一条命令需要 Clang，并用 AddressSanitizer 独立检查 C 世界生成桥接。测试覆盖配置严格校验、名称注册表、26.2 新生物群系、全部结构族、负坐标下界投影、并行搜索顺序、随机全排列的无重复样本、跨线程随机结果一致性、结果数量、CLI JSON 输出和已知种子回归值。

## 第三方代码

项目世界生成核心基于 MIT 许可的 Cubiomes 活跃分支。固定来源、提交和许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `vendor/cubiomes/LICENSE`。
