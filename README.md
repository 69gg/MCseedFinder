# MCSeed Finder

当前面向 Minecraft Java Edition **26.2** 的纯命令行种子筛选器。它可以按出生生物群系、周边生物群系、三维度结构，以及结构内部的具体部件组合筛选种子；例如村庄铁匠铺和各类房屋、沉船变体、埋藏宝藏箱、最近要塞传送门的已有末影之眼或末地船。搜索默认使用多核 CPU，也可以在 Linux x86_64 上选择 CUDA 或 ROCm/HIP 加速。

世界生成核心随项目一起放在 `vendor/cubiomes/`，构建时编译为本地静态代码。程序的构建、运行和测试都不会读取 `sources/`、Minecraft JAR、游戏安装目录或其他被 Git 忽略的输入。

## 构建

需要 Rust 工具链和支持 C11 的 C 编译器：

```bash
cargo build --release
./target/release/mcseed-finder --help
```

第一次构建需要从 crates.io 获取 Rust 通用依赖。Minecraft 世界生成实现已经纳入仓库，不会在构建或运行时下载游戏文件。

默认构建不依赖任何 GPU SDK。Linux x86_64 可以按显卡平台构建一个 GPU 变体；两个 feature 互斥：

```bash
# NVIDIA：需要 CUDA Toolkit（nvcc）
cargo build --release --features cuda

# AMD：需要 ROCm HIP SDK（hipcc）
cargo build --release --features rocm
```

构建脚本会从 `PATH` 查找 `nvcc`/`hipcc`，也接受 `NVCC` 或 `HIPCC` 指定编译器。GPU 变体仍保留完整 CPU 搜索路径；默认 `--accelerator auto` 会在有可下推条件且设备可用时启用已编译的后端，否则说明原因并回退 CPU。完整的构建要求、工作方式和调优建议见 [GPU 加速说明](docs/GPU.md)。

## 常用命令

列出程序实际接受的名称：

```bash
mcseed-finder list biomes
mcseed-finder list structures
mcseed-finder list structures --json
mcseed-finder list pieces --structure village
mcseed-finder list pieces --structure shipwreck --json
```

检查单个种子：

```bash
mcseed-finder check 0 --spawn-biome plains,forest
```

检查出生点 1024 格内的村庄是否包含铁匠铺。现代村庄中的 `blacksmith` 分组包含盔甲匠、武器匠和工具匠房屋：

```bash
mcseed-finder check 0 --piece-near village:blacksmith:1024
```

也可以寻找任一房屋、指定模板、完整沉船或埋藏宝藏箱：

```bash
mcseed-finder find --piece-near village:house:1024 --count 5
mcseed-finder find --piece-near village:village/plains/houses/plains_library_1:1024
mcseed-finder find --piece-near shipwreck:full:4096
mcseed-finder find --piece-near buried_treasure:buried_treasure/chest:4096
```

查询从出生点投掷末影之眼会定位到的最近要塞。`0..12` 只查询并报告，指定精确值或更窄范围即可筛选：

```bash
mcseed-finder check 0 --stronghold-eyes 0..12
mcseed-finder find --random --count 5 --stronghold-eyes 3..12
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

GPU 变体使用 `auto`、`cuda` 或 `rocm`，`hip` 是 `rocm` 的别名；多卡机器可指定设备索引：

```bash
mcseed-finder find \
  --random \
  --count 5 \
  --accelerator rocm \
  --gpu-device 0 \
  --structure-near ruined_portal:256 \
  --piece-near village:blacksmith:512 \
  --progress
```

GPU 会先用共享出生点的结构条件做无需生成出生点的共址必要条件筛选，再批量执行精确出生点估算：每个种子由一个 256 线程工作组处理，气候噪声 octave 和两级候选坐标都在设备上并行计算。出生点锚定条件随后用严格扩大半径粗筛；现代同心环要塞也会在 GPU 上计算 128 个近似位置及安全误差包络。CPU 只为粗筛幸存者继续地表落脚细化，同时 GPU 处理下一批种子，最后再按原半径精筛。CPU 多线程仍负责生物群系定位、地形、Jigsaw 子部件等完整校验，因此不会把近似位置或放置候选直接当成结果。

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
- `--piece-near STRUCTURE:PIECE[,PIECE]:RADIUS`：逗号内为“任一子部件”；`--substructure-near` 是等价别名。半径判断父结构起点，命中同时报告具体部件坐标与父结构起点。
- `--stronghold-eyes EYES|MIN..MAX`：按末影之眼原版定位规则选择距出生点最近的要塞，并筛选传送门框架初始已有眼数；`--end-portal-eyes` 是等价别名。
- 同一种快捷参数可以重复，重复的条件之间为“并且”。
- 顺序模式使用半开区间 `[start, end)`；每批结果会按遍历位置合并，因此不同线程数会返回相同的最小匹配种子。
- `find --random` 使用完整有符号 64 位种子空间的伪随机全排列，不会重复抽取种子。省略 `--random-seed` 时会生成随机化键并立即写到 stderr；使用相同键和条件可以复现相同结果顺序。
- 命令行显式传入 `--random` 时会覆盖配置文件中的 `start`/`end`，便于复用只为顺序搜索编写的条件文件；JSON 自身若设置 `"random": true`，则不能再包含 `start`/`end`。
- `--count N` 指定目标结果数。随机模式省略 `--max-attempts` 时会持续搜索到找满，或在理论上遍历完全部 2^64 个种子；可以随时按 Ctrl+C 停止。
- `--threads N` 控制并行度，默认使用 `available_parallelism` 返回的可用逻辑 CPU 数。工作线程和各自的世界生成上下文会在整次搜索中复用；GPU 模式下它主要影响粗筛幸存者的出生点细化和最终复核，动态分片会让空闲核心继续领取长尾任务。
- `--batch-size` 控制 GPU 批处理、进度刷新和停止检查粒度，默认 4096。更大的值可向 GPU 提供更多独立种子并摊薄固定开销，但会增加内存占用、进度间隔和停止延迟；流水线最多提前计算下一批，但只有真正复核的批次会计入已检查数。
- `--accelerator auto|cpu|cuda|rocm` 控制搜索后端。`auto` 是默认值；显式选择 CUDA/ROCm 时，后端未编译、设备不可用或条件无法安全下推都会直接报错。
- `--gpu-device INDEX` 选择 GPU，默认是 0；它不能和 `--accelerator cpu` 一起使用。活动版本通过 `native/version.h` 声明 GPU 出生点能力，不支持的未来配置会安全回退 CPU，而不是在 Rust 中写死版本字符串。
- `--format jsonl` 适合交给其他程序继续处理。匹配失败时 `check` 和没有结果的 `find` 返回退出码 1，参数或运行错误返回 2。

## JSON 配置

复杂条件使用 JSON。命令行条件会追加到配置中的 `conditions`：

```bash
mcseed-finder find --config examples/village_and_nether.json
mcseed-finder check 0 --config examples/village_and_nether.json --format jsonl
mcseed-finder find --config examples/random_search.json
mcseed-finder find --config examples/village_blacksmith.json
mcseed-finder find --config examples/stronghold_eyes.json
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
    "batch_size": 4096,
    "accelerator": "auto"
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
      "type": "structure_piece_near",
      "structure": "village",
      "any_of": ["blacksmith"],
      "anchor": "spawn",
      "radius": 1024,
      "min_count": 1
    },
    {
      "type": "stronghold_eyes",
      "anchor": "spawn",
      "min_eyes": 3,
      "max_eyes": 12
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

`search.accelerator` 接受 `auto`、`cpu`、`cuda`、`rocm`（以及 `hip` 别名），`search.gpu_device` 是可选的非负设备索引。命令行值优先于 JSON。

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
- `structure_piece_near`（别名 `substructure_near`）：字段 `structure`、`any_of`、`radius`，可选 `anchor`、`min_count`、`max_count`。计数对象是匹配的部件实例，不是父结构数量。
- `stronghold_eyes`：可选 `anchor`、`eyes`、`min_eyes`、`max_eyes`。`eyes` 表示精确值且不能和范围字段并用；范围缺省为 `0..=12`，默认锚点是出生点。
- `all` / `any`：字段 `conditions`，可任意递归组合。
- `not`：字段 `condition`。

`anchor` 可以是 `origin`、`spawn`、`nether_spawn`，也可以是自定义坐标，例如 `{"x": 1200, "z": -300}`。用 `min_count: 0, max_count: 0` 可以表达“范围内不能有该目标”。配置采用严格字段校验，拼错的字段会直接报错。

## 支持范围和精度

`list structures` 列出 26.2 的全部内置结构族：村庄、沙漠神殿、雪屋、丛林神庙、沼泽小屋、掠夺者前哨站、远古城市、海底神殿、林地府邸、埋藏的宝藏、废弃矿井、废弃传送门、沉船、海底废墟、下界要塞、堡垒遗迹、下界化石、末地城、要塞、古迹废墟和试炼密室；废弃传送门按主世界和下界分别提供名称。

- 生物群系范围按 Minecraft 的 4×4×4 噪声生物群系单元扫描；圆形半径按水平方块距离判断。跨 Y 范围会检查所有相交的生物群系单元。
- 出生位置使用 26.2 气候适应度搜索和近似地表检查；出生生物群系会在求得的近似地表 Y 上以方块分辨率读取，避免把地下洞穴群系当成地表出生群系。由于不生成完整方块区块，极少数种子的最终落脚方块仍可能与游戏有少量偏差。
- 大部分结构会执行随机放置、生物群系和已实现的地形可用性检查。沙漠神殿、丛林神庙、林地府邸的地表高度是近似检查；废弃传送门和下界化石还可能因完整区块内部地形而生成失败。`list structures` 会逐项显示精度等级。
- 村庄部件使用从对应版本原版数据提取的模板池权重、模板尺寸和 Jigsaw 连接点，并复现旧式随机源、连接优先级、旋转、碰撞、回退池及最大深度。水平布局与随机序列按 26.2 实现；`WORLD_SURFACE_WG` 使用随项目编译的地形噪声近似，因此极端地形中的 Y 坐标或后续分支可能出现偏差。`list pieces` 会逐项显示这一精度。
- `stronghold_eyes` 会遍历原版同序的 128 个同心环要塞位置，以候选区块中心到锚点的距离选择最近者，再复现传送门房区块装饰随机序列。JSON 命中中的 `eye_count` 是 0..12，`eye_mask` 是低 12 位框架分布；两者表示世界首次生成时的状态，不包含玩家在已有存档中的修改。
- 这里筛选的是默认世界类型；不支持大型生物群系、数据包改写的世界生成或自定义维度。

当前可查询的子部件范围：

- 村庄：`blacksmith`、`house`、`profession_house`、`residential`、全部村民职业（同时接受 `librarian`/`library`、`cleric`/`temple` 等职业名和建筑名）、农田、畜栏、马厩、大小房屋、集会点、街道、装饰、僵尸村庄等分组，以及 491 个完整模板/特征名称。
- 沉船：完整、前半、后半、桅杆、完好、破损分组及全部 20 个模板变体；埋藏宝藏可查 `buried_treasure/chest`。
- 下界要塞、要塞和末地城：底层提供的全部部件类型，并提供 `blaze_spawner`、`nether_wart_room`、`portal_room`、`library`、`ship` 等常用分组。
- 雪屋、沙漠神殿、丛林神庙、沼泽小屋，以及主世界/下界废弃传送门的已知模板均可查询。
- 掠夺者前哨站目前只提供瞭望塔；堡垒遗迹目前只提供底层稳定实现的四个必有部件，二者在列表中标为 `partial`。矿井、林地府邸、海底神殿、海底废墟、远古城市、古迹废墟、试炼密室和下界化石目前仅支持父结构搜索。

名称不需要猜：用 `list pieces --structure STRUCTURE` 获取程序实际接受的分组和精确名称。

GPU 预筛选覆盖注册表中采用随机分区放置的 20 个结构族，并额外覆盖现代版本的同心环要塞；废弃矿井继续由 CPU 处理。要塞 GPU 阶段只计算原版近似环，并以生物群系搜索半径和设备数学舍入余量扩张包络，幸存者仍由 Cubiomes 精确定位。程序会自动提取必须成立的正向 `structure_near` 条件、正向 `structure_piece_near` 所必需的父结构候选，以及共享同一出生点锚点的随机分区结构对所必须满足的共址关系。`any`、`not`、纯上限/排除条件不会被不安全地下推；不能加速的条件始终留在 CPU 精确路径中。

如果要在游戏中长期使用找到的种子，建议先用 `check` 保存见证坐标，再在原版 26.2 中做一次最终确认。

## 测试

```bash
cargo test --offline
cargo clippy --all-targets --offline -- -D warnings
sh scripts/test-native-sanitizer.sh

# 安装对应 SDK 后额外验证 GPU 变体
cargo test --offline --features cuda
cargo test --offline --features rocm
```

`scripts/test-native-sanitizer.sh` 需要 Clang，并用 AddressSanitizer 与 UndefinedBehaviorSanitizer 独立检查 C 世界生成桥接和共享的 GPU 放置参考实现；默认同时启用 LeakSanitizer。若运行环境处于 ptrace 下而无法启动 LSan，可设置 `MCSEED_ASAN_DETECT_LEAKS=0`，地址与未定义行为检查仍会执行。测试覆盖配置严格校验、名称注册表、村庄铁匠铺/房屋的确定性布局、沉船与宝藏部件、最近要塞的眼框掩码、强要塞剪枝与全量参考逐坐标一致性、26.2 生物群系树索引边界与新增生物群系、CPU/GPU 出生点逐种子一致性、流水线掩码顺序、动态分片无遗漏/无重复、全部结构族、结构对共址预筛选、负坐标下界投影、并行搜索顺序、随机全排列、跨线程和批大小结果一致性、GPU 保守性与设备内核一致性、CLI JSON 输出和已知种子回归值。

## 扩展 Minecraft 版本

当前版本配置集中在 `native/version.h`；Rust 输出版本、GPU 出生点算法和强要塞同心环参数都从这里读取。村庄数据按版本放在 `native/generated/`，Jigsaw 查询接口本身接收版本标签。新增版本时，应先验证 Cubiomes 对应版本，再显式选择兼容的 GPU 能力配置，并从合法取得的客户端 JAR 生成紧凑元数据：

```bash
python3 tools/extract_jigsaw.py \
  --jar /path/to/client.jar \
  --version 26.2 \
  --output native/generated/village_26_2.inc
```

生成文件只包含池权重、尺寸和连接点；将它纳入 `native/jigsaw.c` 的版本注册表后提交。提取脚本和 JAR 都不会在构建、测试或运行时执行/读取。

## 第三方代码

项目世界生成核心基于 MIT 许可的 Cubiomes 活跃分支。固定来源、提交和许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `vendor/cubiomes/LICENSE`。
