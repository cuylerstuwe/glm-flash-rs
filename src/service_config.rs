//! Explicit, pinned configuration for an isolated native service. Credentials
//! are separate private files and never serialized into status or evidence.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Read,
    net::SocketAddr,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Transport {
    Roce,
    Nccl,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Grammar {
    pub library: PathBuf,
    pub sha256: String,
}
impl Grammar {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.library.is_absolute()
                && !self
                    .library
                    .components()
                    .any(|p| matches!(p, std::path::Component::ParentDir)),
            "grammar library path"
        );
        ensure!(
            self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "grammar library pin encoding"
        );
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Vision {
    pub socket: PathBuf,
    #[serde(default = "default_image_fetch")]
    pub fetch: bool,
}
fn default_image_fetch() -> bool {
    true
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Speculation {
    pub manifest: PathBuf,
    pub costs: crate::depth_policy::Costs,
    #[serde(default)]
    pub lookup: bool,
}
impl Speculation {
    pub(crate) fn digest(&self) -> Result<[u8; 32]> {
        self.costs.validate()?;
        let mut h = Sha256::new();
        h.update(b"service-speculation-v1");
        h.update([u8::from(self.lookup)]);
        h.update(serde_json::to_vec(&self.costs)?);
        Ok(h.finalize().into())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub version: u32,
    pub experimental: bool,
    #[serde(default)]
    pub qualification: Option<crate::production_qualification::Qualification>,
    pub rank: usize,
    pub manifest: PathBuf,
    pub kernels: PathBuf,
    pub native: PathBuf,
    pub nccl: PathBuf,
    pub master: SocketAddr,
    pub transport: Transport,
    pub binary_sha256: String,
    pub state: PathBuf,
    pub tokenizer: PathBuf,
    pub listen: SocketAddr,
    /// Absent: inference routes need no API key.
    #[serde(default)]
    pub api_key_file: Option<PathBuf>,
    pub scheduler_key_file: PathBuf,
    /// Permit a non-loopback `listen` address (the production endpoint).
    #[serde(default)]
    pub public_listener: bool,
    pub served_model: String,
    pub ram_bytes: u64,
    pub ssd_bytes: u64,
    pub thinking: bool,
    pub reasoning_effort: String,
    pub clear_thinking_file: PathBuf,
    pub clear_thinking_default: Option<bool>,
    /// Whole-request deadline, including preparation and deferred grammar work.
    /// Defaults to one hour when omitted.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "default_prefill_tokens")]
    pub prefill_tokens: usize,
    /// Compute the output head only for the final prefill slab when enabled.
    /// Disabled by default; qualify model/cache/priority behavior before enabling.
    #[serde(default)]
    pub final_prefill_head_only: bool,
    /// Batch the serial-exact leading fragment of an off-grid mixed prefix
    /// through the already row-invariant target verifier. Qualification opt-in.
    #[serde(default)]
    pub exact_prompt_batch: bool,
    #[serde(default)]
    pub speculation: Option<Speculation>,
    /// Explicit qualification candidate: use serial-exact greedy target
    /// verification for temperature <= 0 requests. Default remains serial.
    #[serde(default)]
    pub greedy_speculation: bool,
    #[serde(default)]
    pub grammar: Option<Grammar>,
    /// Rank-zero vision encoder sidecar (`vision-serve`) socket; images are
    /// refused without it.
    #[serde(default)]
    pub vision: Option<Vision>,
    // Full-logit diagnostic readbacks are enabled by default. Serving setups
    // disable them explicitly after qualifying model and cache behavior.
    #[serde(default = "default_audit_logits")]
    pub audit_logits: bool,
}
fn default_audit_logits() -> bool {
    true
}
fn default_prefill_tokens() -> usize {
    512
}
fn default_request_timeout_ms() -> u64 {
    3_600_000
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let f = regular(path, false).context("opening service configuration")?;
        ensure!(f.metadata()?.len() <= 16384, "service configuration size");
        let value: serde_json::Value =
            serde_json::from_reader(f.take(16385)).context("service configuration JSON")?;
        let config: Self = serde_json::from_value(value.clone())?;
        config.validate()?;
        if let Some(qualification) = &config.qualification {
            qualification.deployment(&config.binary_sha256, config.rank, &value)?;
        }
        let mut executable = File::open(std::env::current_exe()?)?;
        let mut hash = Sha256::new();
        let mut block = [0; 64 << 10];
        loop {
            let n = executable.read(&mut block)?;
            if n == 0 {
                break;
            }
            hash.update(&block[..n]);
        }
        ensure!(
            format!("{:x}", hash.finalize()) == config.binary_sha256,
            "service executable differs from deployment pin"
        );
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "service configuration version");
        if self.experimental {
            ensure!(
                self.qualification.is_none(),
                "experimental config cannot advertise qualification"
            );
        } else {
            self.qualification
                .as_ref()
                .context("production requires a pinned qualification report")?
                .validate(&self.binary_sha256)?;
        }
        ensure!(self.rank < 4, "service rank");
        ensure!(
            (1..=86_400_000).contains(&self.request_timeout_ms),
            "service request timeout bounds"
        );
        if let Some(grammar) = &self.grammar {
            grammar.validate()?;
        }
        if let Some(vision) = &self.vision {
            ensure!(vision.socket.is_absolute(), "vision socket path");
        }
        ensure!(
            !self.greedy_speculation || self.speculation.is_some(),
            "greedy speculation requires a speculative owner"
        );
        ensure!(
            !self.exact_prompt_batch || self.speculation.is_some(),
            "exact prompt batching requires a speculative owner"
        );
        if let Some(speculation) = &self.speculation {
            speculation.costs.validate()?;
            ensure!(
                speculation.costs.verify.len() == if speculation.lookup { 16 } else { 8 },
                "speculative measured cost width"
            );
            ensure!(
                speculation.manifest.is_absolute()
                    && !speculation
                        .manifest
                        .components()
                        .any(|p| matches!(p, std::path::Component::ParentDir)),
                "drafter manifest path"
            );
        }
        ensure!(
            matches!(self.prefill_tokens, 512 | 8192),
            "service prefill geometry"
        );
        ensure!(
            self.binary_sha256.len() == 64
                && self
                    .binary_sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "binary pin encoding"
        );
        ensure!(
            self.listen.port() != 0
                && (self.listen.ip().is_loopback() || (self.public_listener && self.rank == 0)),
            "service listener must be a fixed loopback address unless rank 0 is public"
        );
        ensure!(
            self.master.port() != 0 && !self.master.ip().is_unspecified(),
            "collective address"
        );
        ensure!(
            ((32u64 << 20)..=(16u64 << 30)).contains(&self.ram_bytes)
                && ((32u64 << 20)..=(64u64 << 30)).contains(&self.ssd_bytes),
            "service cache budgets"
        );
        ensure!(
            !self.served_model.is_empty()
                && self.served_model.len() <= 160
                && !self.served_model.chars().any(char::is_control),
            "served model name"
        );
        ensure!(
            matches!(self.reasoning_effort.as_str(), "low" | "high"),
            "reasoning effort"
        );
        for path in [
            &self.manifest,
            &self.kernels,
            &self.native,
            &self.nccl,
            &self.state,
            &self.tokenizer,
            &self.scheduler_key_file,
            &self.clear_thinking_file,
        ]
        .into_iter()
        .chain(self.api_key_file.as_ref())
        {
            ensure!(
                path.is_absolute()
                    && !path
                        .components()
                        .any(|p| matches!(p, std::path::Component::ParentDir)),
                "service paths must be absolute and normalized"
            );
        }
        Ok(())
    }
}
pub(crate) fn secret(path: &Path) -> Result<String> {
    let f = regular(path, true).context("opening private service key")?;
    let m = f.metadata()?;
    ensure!(
        m.is_file()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o077 == 0
            && (16..=514).contains(&m.len()),
        "service key file owner/mode/size"
    );
    let mut value = String::new();
    f.take(515).read_to_string(&mut value)?;
    let value = value.trim_end_matches(['\r', '\n']);
    ensure!(
        (16..=512).contains(&value.len())
            && !value.chars().any(char::is_whitespace)
            && !value.chars().any(char::is_control),
        "service key format"
    );
    Ok(value.to_owned())
}

fn regular(path: &Path, private: bool) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)?;
    let m = file.metadata()?;
    ensure!(
        m.is_file()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & if private { 0o077 } else { 0o022 } == 0,
        "service file owner/type/mode"
    );
    Ok(file)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    #[test]
    fn speculative_configuration_is_explicit_bounded_and_defaults_to_serial_audit() -> Result<()> {
        let base = serde_json::json!({
            "version":1,"experimental":true,"rank":0,"manifest":"/model/manifest",
            "kernels":"/kernels","native":"/native","nccl":"/nccl","master":"127.0.0.1:2000",
            "transport":"roce","binary_sha256":"0".repeat(64),"state":"/state","tokenizer":"/tokenizer",
            "listen":"127.0.0.1:8000","api_key_file":"/key","scheduler_key_file":"/control",
            "served_model":"qualification","ram_bytes":33554432,"ssd_bytes":33554432,"thinking":true,
            "reasoning_effort":"high","clear_thinking_file":"/history","clear_thinking_default":null
        });
        let config: Config = serde_json::from_value(base.clone())?;
        config.validate()?;
        assert_eq!(config.request_timeout_ms, 3_600_000);
        for timeout in [1, 1500, 3_600_000, 86_400_000] {
            let mut value = base.clone();
            value["request_timeout_ms"] = timeout.into();
            let config: Config = serde_json::from_value(value)?;
            config.validate()?;
            assert_eq!(config.request_timeout_ms, timeout);
        }
        for timeout in [0, 86_400_001, u64::MAX] {
            let mut value = base.clone();
            value["request_timeout_ms"] = timeout.into();
            assert!(serde_json::from_value::<Config>(value)?.validate().is_err());
        }
        for (rank, public, ok) in [(0, true, true), (0, false, false), (1, true, false)] {
            let mut value = base.clone();
            value["rank"] = rank.into();
            value["listen"] = "0.0.0.0:8001".into();
            value["public_listener"] = public.into();
            assert_eq!(
                serde_json::from_value::<Config>(value)?.validate().is_ok(),
                ok
            );
        }
        assert!(
            config.speculation.is_none()
                && !config.greedy_speculation
                && config.grammar.is_none()
                && config.audit_logits
                && !config.final_prefill_head_only
                && !config.exact_prompt_batch
        );
        let mut orphan = base.clone();
        orphan["exact_prompt_batch"] = true.into();
        assert!(
            serde_json::from_value::<Config>(orphan.clone())?
                .validate()
                .is_err()
        );
        orphan["exact_prompt_batch"] = false.into();
        orphan["greedy_speculation"] = true.into();
        assert!(
            serde_json::from_value::<Config>(orphan)?
                .validate()
                .is_err()
        );
        let mut grammar = base.clone();
        grammar["grammar"] =
            serde_json::json!({"library":"/native/grammar.so","sha256":"a".repeat(64)});
        serde_json::from_value::<Config>(grammar.clone())?.validate()?;
        for path in ["relative", "/native/../grammar.so"] {
            let mut bad = grammar.clone();
            bad["grammar"]["library"] = path.into();
            assert!(serde_json::from_value::<Config>(bad)?.validate().is_err());
        }
        for pin in [
            "a".repeat(63),
            "a".repeat(65),
            "A".repeat(64),
            "g".repeat(64),
        ] {
            let mut bad = grammar.clone();
            bad["grammar"]["sha256"] = pin.into();
            assert!(serde_json::from_value::<Config>(bad)?.validate().is_err());
        }
        for lookup in [false, true] {
            let mut value = base.clone();
            value["speculation"] = serde_json::json!({"manifest":"/draft/manifest","lookup":lookup,
                "costs":{"verify":vec![1.;if lookup {16} else {8}],"block":1.,"taps_row":0.,"round":0.}});
            let config: Config = serde_json::from_value(value.clone())?;
            config.validate()?;
            assert!(!config.greedy_speculation);
            let mut batched = value.clone();
            batched["exact_prompt_batch"] = true.into();
            let batched: Config = serde_json::from_value(batched)?;
            batched.validate()?;
            assert!(batched.exact_prompt_batch);
            let mut greedy = value.clone();
            greedy["greedy_speculation"] = true.into();
            let greedy: Config = serde_json::from_value(greedy)?;
            greedy.validate()?;
            assert!(greedy.greedy_speculation);
            let expected = config.speculation.unwrap().digest()?;
            value["speculation"]["costs"]["verify"][0] = serde_json::json!(2.);
            let config: Config = serde_json::from_value(value.clone())?;
            assert_ne!(config.speculation.unwrap().digest()?, expected);
            for path in ["relative", "/draft/../manifest"] {
                let mut bad = value.clone();
                bad["speculation"]["manifest"] = path.into();
                assert!(serde_json::from_value::<Config>(bad)?.validate().is_err());
            }
            for costs in [
                serde_json::json!([1.]),
                serde_json::json!(vec![0.; 8]),
                serde_json::json!(vec![1.; 9]),
            ] {
                let mut bad = value.clone();
                bad["speculation"]["costs"]["verify"] = costs;
                assert!(serde_json::from_value::<Config>(bad)?.validate().is_err());
            }
        }
        Ok(())
    }
    #[test]
    fn private_keys_reject_links_fifo_permissions_and_oversize() -> Result<()> {
        let root = std::env::temp_dir().join(format!("alaska-service-keys-{}", std::process::id()));
        std::fs::create_dir(&root)?;
        let key = root.join("key");
        std::fs::write(&key, "test-key-with-enough-bytes\n")?;
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600))?;
        assert_eq!(secret(&key)?, "test-key-with-enough-bytes");
        let link = root.join("link");
        symlink(&key, &link)?;
        assert!(secret(&link).is_err());
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644))?;
        assert!(secret(&key).is_err());
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600))?;
        std::fs::write(&key, vec![b'x'; 515])?;
        assert!(secret(&key).is_err());
        let fifo = root.join("fifo");
        let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes())?;
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        assert!(secret(&fifo).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
