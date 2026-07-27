use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::accelerator::AcceleratorKind;

#[derive(Debug, Parser)]
#[command(
    name = "mcseed-finder",
    version,
    about = "Minecraft Java 种子筛选器",
    long_about = "按出生生物群系、指定半径内的生物群系、主世界/下界/末地结构及结构内部部件搜索 Minecraft Java 种子。"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// 按顺序或随机顺序查找匹配项
    Find(FindArgs),
    /// 检查一个种子并输出每个条件的结果
    Check(CheckArgs),
    /// 列出可用的生物群系、结构或子部件名称
    List(ListArgs),
}

#[derive(Debug, Args)]
pub struct FindArgs {
    #[command(flatten)]
    pub filters: FilterArgs,

    /// 起始种子（包含）
    #[arg(long, allow_hyphen_values = true)]
    pub start: Option<i128>,

    /// 结束种子（不包含）；可设为 9223372036854775808 以包含 i64::MAX
    #[arg(long, allow_hyphen_values = true)]
    pub end: Option<i128>,

    /// 在完整 64 位种子空间中以无重复随机顺序查找，默认持续到找满
    #[arg(long)]
    pub random: bool,

    /// 随机顺序的可复现键；省略时自动生成并输出到 stderr
    #[arg(long, value_name = "U64")]
    pub random_seed: Option<u64>,

    /// 随机模式最多检查多少个种子；省略时可遍历完整 64 位空间
    #[arg(long, value_name = "N")]
    pub max_attempts: Option<u128>,

    /// 目标输出多少个匹配种子
    #[arg(short = 'n', long = "count")]
    pub results: Option<usize>,

    /// 并行工作线程数；默认使用系统报告的可用逻辑 CPU 数
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// 每批按当前遍历顺序派发的种子数量
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// 搜索后端：auto、cpu、cuda 或 rocm（hip 是 rocm 的别名）
    #[arg(long, value_enum)]
    pub accelerator: Option<AcceleratorKind>,

    /// CUDA/ROCm 设备索引；默认使用设备 0
    #[arg(long, value_name = "INDEX")]
    pub gpu_device: Option<u32>,

    /// 输出格式
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    /// 在 stderr 显示搜索速度
    #[arg(long)]
    pub progress: bool,
}

#[derive(Debug, Args)]
pub struct CheckArgs {
    /// Minecraft 有符号 64 位种子
    #[arg(allow_hyphen_values = true)]
    pub seed: i64,

    #[command(flatten)]
    pub filters: FilterArgs,

    /// 输出格式
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Args)]
pub struct FilterArgs {
    /// JSON 配置文件；命令行条件会追加到文件条件中
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// 允许的出生生物群系，逗号表示“任一”；可重复
    #[arg(long, value_name = "BIOME[,BIOME]")]
    pub spawn_biome: Vec<String>,

    /// DIMENSION:BIOME[,BIOME]:RADIUS[:Y|Y_MIN..Y_MAX]；末地默认锚定首个折跃门外岛出口
    #[arg(long, value_name = "SPEC")]
    pub biome_near: Vec<String>,

    /// STRUCTURE[,STRUCTURE]:RADIUS；末地默认锚定首个折跃门外岛出口，可重复
    #[arg(long, value_name = "SPEC")]
    pub structure_near: Vec<String>,

    /// STRUCTURE:PIECE[,PIECE]:RADIUS；末地默认锚定首个折跃门外岛出口，可重复
    #[arg(long, alias = "substructure-near", value_name = "SPEC")]
    pub piece_near: Vec<String>,

    /// 最近要塞传送门的已有眼数；例如 3 或 3..12，可重复
    #[arg(long, alias = "end-portal-eyes", value_name = "EYES|MIN..MAX")]
    pub stronghold_eyes: Vec<String>,
}

#[derive(Debug, Args)]
pub struct ListArgs {
    #[arg(value_enum)]
    pub kind: ListKind,

    /// 输出 JSON
    #[arg(long)]
    pub json: bool,

    /// 仅列出指定父结构的子结构；只用于 pieces
    #[arg(long, value_name = "STRUCTURE")]
    pub structure: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ListKind {
    Biomes,
    Structures,
    Pieces,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Text,
    Jsonl,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};
    use crate::accelerator::AcceleratorKind;

    #[test]
    fn hip_cli_alias_selects_the_rocm_backend() {
        let cli = Cli::try_parse_from(["mcseed-finder", "find", "--accelerator", "hip"])
            .expect("hip 应是合法的 ROCm CLI 别名");
        let Command::Find(arguments) = cli.command else {
            panic!("应解析为 find 子命令");
        };
        assert_eq!(arguments.accelerator, Some(AcceleratorKind::Rocm));
    }
}
