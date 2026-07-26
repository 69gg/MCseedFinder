# GPU 加速

MCSeed Finder 的 GPU 后端面向 Linux x86_64，支持 NVIDIA CUDA 和 AMD ROCm/HIP。默认 CPU 构建没有 GPU 运行时依赖；CUDA 与 ROCm 是互斥的可选 Cargo feature。

## 构建

CUDA 构建需要可用的 CUDA Toolkit 和 `nvcc`：

```bash
cargo build --release --features cuda
```

构建脚本从 `PATH` 查找 `nvcc`。非标准安装可以设置 `NVCC=/path/to/nvcc`。程序运行时还需要与该 Toolkit 兼容的 NVIDIA 驱动。

ROCm 构建需要 ROCm HIP SDK 和 `hipcc`：

```bash
cargo build --release --features rocm
```

构建脚本依次使用 `HIPCC`、`PATH` 定位编译器，并通过 `ROCM_PATH`、`HIP_PATH`、`hipconfig --path` 或编译器位置发现 `libamdhip64`。例如：

```bash
HIPCC=/opt/rocm/bin/hipcc cargo build --release --features rocm
```

当前若为非 Linux x86_64 目标，启用 GPU feature 会在构建期给出明确错误；不启用 feature 的 CPU 版本不受这个限制。不要同时启用 `cuda,rocm`。

## 使用

GPU 变体默认采用自动模式：

```bash
mcseed-finder find \
  --random \
  --count 10 \
  --accelerator auto \
  --structure-near village:256
```

自动模式只有在以下条件都成立时才启用 GPU：

- 二进制包含 CUDA 或 ROCm 后端；
- 运行时检测到设备；
- 查询包含至少一个可安全下推的结构放置必要条件。

否则它会在 stderr 说明原因并回退多线程 CPU。需要严格要求某个后端时，显式使用 `--accelerator cuda` 或 `--accelerator rocm`；这时不会静默回退。`--gpu-device N` 选择设备，默认是 0。

JSON 对应字段位于 `search`：

```json
{
  "search": {
    "accelerator": "rocm",
    "gpu_device": 0,
    "threads": 12,
    "batch_size": 4096
  }
}
```

`hip` 也可作为 `rocm` 的输入别名。

## 正确性模型

GPU 执行的是保守预筛选，而不是另一套简化世界生成器：

1. CPU 按遍历顺序生成种子。对于共享同一出生点锚点的两个正向结构条件，GPU 先检查是否存在距离不超过两条件半径之和的候选结构对。这是不需要生成出生点的必要条件，不满足时可以立即淘汰种子。
2. 结构对扫描范围使用出生算法的严格原点包络：主世界出生点不超过原点 2697 格，下界投影不超过 340 格；再分别加上条件半径，因此不会因扫描窗口过小漏解。
3. 多个 CPU 工作线程只为幸存种子调用较便宜的 `estimateSpawn`。当前现代出生算法保证估计点到最终点不超过 125 格；GPU 用相应扩大后的半径做第一轮保守粗筛。下界投影会把这个上界安全换算为 18 格。
4. CPU 从已保存的估计点继续执行带地表落脚检查的完整出生点细化，不会再次运行估算搜索。GPU 随后用原始半径精筛；固定坐标条件不需要这一级出生点准备。
5. 最终幸存种子回到原有 Cubiomes 路径，完整检查生物群系、地形、结构竞争、Jigsaw 布局、末地要塞眼数等条件。
6. 只有 CPU 精确路径通过的种子才会输出；完整出生点会在报告阶段复用，不会再次生成。

出生点估算本身保持 Mojang 26.2 的候选顺序和单精度角度/半径推进方式，但会逐级计算严格下界：先检查到原点的距离，再检查偏移与怪异度、陆地性，只有仍可能优于当前最优解时才采样其余气候噪声。完整适应度会复用这些已采样值。该优化只省略不可能获胜的计算，不改变相等值的先后选择规则；回归测试会逐种子与未优化参考实现比较坐标。

放置参数来自 `native/bridge.c` 的现有结构注册表与 Cubiomes 版本配置。CPU 参考实现和 CUDA/HIP 内核共享 `native/gpu/placement.h`。原生测试会逐分区比较全部白名单结构族的共享算法与 Cubiomes 放置结果，并验证参考预筛选不会漏掉精确结构命中；装有 GPU SDK/设备时还会逐字节比较设备内核与 CPU 参考掩码。

结构放置算法采用显式白名单映射；以后新增但尚未确认放置语义的结构默认留在 CPU 路径，不会猜测其兼容性。

当前可下推 22 个结构族中的 20 个随机分区放置族。废弃矿井和同心环要塞不使用该模型，因此不下推。以下逻辑也会保留在 CPU：

- 生物群系、完整地形和出生生物群系；
- 最近要塞与末地传送门已有眼数；
- Jigsaw 或其他结构内部部件的最终判断；
- `any`、`not`，以及只有最大数量的排除条件。

正向子部件条件可以安全下推“父结构候选必须存在”这一必要条件，但部件本身仍由 CPU 生成并验证。若一个“任一结构”条件中含有无法下推的结构族，整个条件不会部分下推，以防漏解。

## 性能

GPU 对半径较小、放置候选稀少、后续生物群系或子结构校验昂贵的组合最有效。只有出生生物群系、要塞眼数，或半径大到几乎每个种子都有候选时，GPU 可能没有收益；自动模式会跳过完全不可加速的查询，但不会尝试根据未知硬件自动预测盈亏。

同一查询的结构对预筛、粗筛和精筛会分别按扫描面积与估计放置密度排序，让更可能淘汰候选的条件先执行并尽早结束当前线程；这只重排必须同时成立的必要条件，不影响筛选语义。

`--threads` 在 GPU 模式下仍控制 CPU 出生点估算、幸存者细化和精确复核并行度，通常保留系统默认值即可。出生点准备、估算与细化会把批次拆成多于线程数的小分片并通过共享队列动态领取，少量候选搜索路径较长时，已空闲的核心可以继续处理剩余分片。

`--batch-size` 默认 4096；更大的批次可摊薄很便宜查询的固定开销，但会增加内存占用、进度刷新间隔和找到足够结果后的停止延迟。出生点 CPU 工作会在批次内部动态均衡，因此大批次不再依赖每个静态线程分片具有相同成本；仍建议从默认值开始，按实际硬件测量。

进度与完成摘要会分别显示无需出生点的结构对预筛、出生点估算、扩大半径粗筛、出生点细化、原半径精筛和 CPU 完整验证耗时，并报告各级保留数/原始候选数，便于直接判断真正的瓶颈和筛选选择性。

## 验证

默认 CPU 路径：

```bash
cargo test --offline
cargo clippy --all-targets --offline -- -D warnings
```

对应 GPU SDK 已安装时：

```bash
cargo test --offline --features cuda
cargo clippy --all-targets --offline --features cuda -- -D warnings

cargo test --offline --features rocm
cargo clippy --all-targets --offline --features rocm -- -D warnings
```

GPU 测试只使用固定的小样本，并比较 CPU/GPU 结果；不会启动大范围随机搜索。
