use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dimension {
    Overworld,
    Nether,
    End,
}

impl Dimension {
    pub const fn as_raw(self) -> i32 {
        match self {
            Self::Overworld => 0,
            Self::Nether => -1,
            Self::End => 1,
        }
    }

    pub fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Overworld),
            -1 => Some(Self::Nether),
            1 => Some(Self::End),
            _ => None,
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Overworld => "overworld",
            Self::Nether => "nether",
            Self::End => "end",
        };
        formatter.write_str(value)
    }
}

impl std::str::FromStr for Dimension {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "overworld" | "主世界" => Ok(Self::Overworld),
            "nether" | "下界" => Ok(Self::Nether),
            "end" | "末地" => Ok(Self::End),
            _ => Err(format!(
                "未知维度 {value:?}，可用值：overworld、nether、end"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Position {
    pub x: i32,
    pub y: Option<i32>,
    pub z: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpawnInfo {
    pub position: Position,
    pub biome: String,
    #[serde(skip)]
    pub biome_id: i32,
}

pub fn project_to_nether(value: i32) -> i32 {
    value.div_euclid(8)
}

#[cfg(test)]
mod tests {
    use super::project_to_nether;

    #[test]
    fn nether_projection_uses_floor_division() {
        assert_eq!(project_to_nether(15), 1);
        assert_eq!(project_to_nether(8), 1);
        assert_eq!(project_to_nether(7), 0);
        assert_eq!(project_to_nether(0), 0);
        assert_eq!(project_to_nether(-1), -1);
        assert_eq!(project_to_nether(-8), -1);
        assert_eq!(project_to_nether(-9), -2);
    }
}
