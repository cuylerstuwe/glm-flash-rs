//! Process-lifetime service lease and bounded live history policy.
use anyhow::{Context, Result, ensure};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::OnceLock,
};
static LEASE: OnceLock<File> = OnceLock::new();
pub(crate) fn lease(root: &Path) -> Result<()> {
    if !root.exists() {
        fs::create_dir(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
    }
    let m = fs::symlink_metadata(root)?;
    ensure!(
        m.is_dir() && m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "service state requires a private owned directory"
    );
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(root.join(".service-lock"))?;
    ensure!(f.metadata()?.is_file(), "service lock type");
    f.try_lock().context("service state already owned")?;
    // Intentionally process-lifetime: a detached, retiring file worker cannot
    // outlive this lease and race a replacement process during Rust unwinding.
    LEASE
        .set(f)
        .map_err(|_| anyhow::anyhow!("service lease already installed"))?;
    // Publish PID only after lease acquisition, so a duplicate launch cannot
    // overwrite the actual owner's cleanup identity.
    let temporary = root.join(".pid.next");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(&temporary)?;
    ensure!(file.metadata()?.is_file(), "PID publication file type");
    use std::io::Write;
    writeln!(file, "{}", std::process::id())?;
    file.sync_all()?;
    fs::rename(temporary, root.join("pid"))?;
    Ok(())
}
pub(crate) struct History {
    path: PathBuf,
    default: Option<bool>,
}
impl History {
    pub fn new(path: PathBuf, default: Option<bool>) -> Self {
        Self { path, default }
    }
    pub fn current(&self) -> Option<bool> {
        self.read().ok().flatten().or(self.default)
    }
    fn read(&self) -> Result<Option<bool>> {
        let f = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(&self.path)?;
        ensure!(
            f.metadata()?.is_file() && f.metadata()?.len() <= 256,
            "history policy size/type"
        );
        let mut text = String::new();
        f.take(257).read_to_string(&mut text)?;
        ensure!(text.len() <= 256, "history policy limit");
        parse(&text)
    }
}
fn parse(text: &str) -> Result<Option<bool>> {
    Ok(match text.trim().to_ascii_lowercase().as_str() {
        "" | "default" => None,
        "keep" | "preserve" | "preserved" | "0" | "false" | "no" | "off" => Some(false),
        "clear" | "1" | "true" | "yes" | "on" => Some(true),
        _ => anyhow::bail!("invalid history policy"),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_history_tracks_changes_and_falls_back() -> Result<()> {
        let file = std::env::temp_dir().join(format!("alaska-history-{}", std::process::id()));
        let h = History::new(file.clone(), Some(false));
        assert_eq!(h.current(), Some(false));
        fs::write(&file, " CLEAR\n")?;
        assert_eq!(h.current(), Some(true));
        fs::write(&file, "default")?;
        assert_eq!(h.current(), Some(false));
        fs::write(&file, "invalid")?;
        assert_eq!(h.current(), Some(false));
        fs::remove_file(file)?;
        assert_eq!(h.current(), Some(false));
        for s in ["keep", "preserve", "preserved", "0", "false", "no", "off"] {
            assert_eq!(parse(s)?, Some(false));
        }
        Ok(())
    }
}
