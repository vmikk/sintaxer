use anyhow::{Result, bail, ensure};
use std::collections::BTreeMap;

pub const RANKS: &[u8] = b"dkpcofg";
pub type Lineage = [Option<String>; 7];

pub fn parse(header: &str) -> Result<Lineage> {
    let (_, rest) = header
        .split_once(";tax=")
        .ok_or_else(|| anyhow::anyhow!("missing ;tax= annotation"))?;
    ensure!(!rest.contains(";tax="), "multiple tax= annotations");
    let tax = rest.split(';').next().unwrap();
    let mut lineage: Lineage = Default::default();
    let mut previous = None;
    for item in tax.split(',') {
        let bytes = item.as_bytes();
        ensure!(
            bytes.len() >= 3 && bytes[1] == b':',
            "invalid rank annotation {item:?}"
        );
        let rank = b"dkpcofgst"
            .iter()
            .position(|r| *r == bytes[0])
            .ok_or_else(|| anyhow::anyhow!("unknown rank in {item:?}"))?;
        ensure!(
            previous.is_none_or(|p| rank > p),
            "ranks must be unique and ordered"
        );
        previous = Some(rank);
        let name = &item[2..];
        ensure!(
            !name.trim().is_empty() && !name.chars().any(char::is_control),
            "empty or invalid taxon name"
        );
        if rank < 7 {
            lineage[rank] = Some(name.to_owned());
        }
    }
    if lineage.iter().all(Option::is_none) {
        bail!("at least one rank through genus is required");
    }
    Ok(lineage)
}

#[derive(Clone, Debug)]
pub struct Node {
    pub parent: u32,
    pub rank: u8,
    pub name: String,
}

pub struct TreeBuilder {
    pub nodes: Vec<Node>,
    lookup: BTreeMap<(u32, u8, String), u32>,
}
impl Default for TreeBuilder {
    fn default() -> Self {
        Self {
            nodes: vec![Node {
                parent: 0,
                rank: 255,
                name: String::new(),
            }],
            lookup: BTreeMap::new(),
        }
    }
}
impl TreeBuilder {
    pub fn insert(&mut self, lineage: Lineage) -> Result<u32> {
        let mut parent = 0;
        for (rank, name) in lineage.into_iter().enumerate() {
            let name = name.unwrap_or_default();
            let key = (parent, rank as u8, name.clone());
            parent = if let Some(&id) = self.lookup.get(&key) {
                id
            } else {
                let id = u32::try_from(self.nodes.len())?;
                self.nodes.push(Node {
                    parent,
                    rank: rank as u8,
                    name,
                });
                self.lookup.insert(key, id);
                id
            };
        }
        Ok(parent)
    }
}
