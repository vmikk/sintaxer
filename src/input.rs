use anyhow::{Context, Result, ensure};
use needletail::FastxReader;
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

pub fn reader(path: &Path) -> Result<Option<Box<dyn FastxReader>>> {
    let input: Box<dyn Read + Send> = if path.as_os_str() == "-" {
        Box::new(io::stdin())
    } else {
        Box::new(File::open(path).with_context(|| format!("opening {}", path.display()))?)
    };
    match needletail::parse_fastx_reader(input) {
        Ok(reader) => Ok(Some(reader)),
        Err(error) if error.kind == needletail::errors::ParseErrorKind::EmptyFile => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub fn label(id: &[u8]) -> Result<String> {
    let label = std::str::from_utf8(id).context("sequence label must be UTF-8")?;
    ensure!(
        !label.is_empty() && !label.chars().any(char::is_control),
        "empty label or control character in label"
    );
    Ok(label.to_owned())
}
