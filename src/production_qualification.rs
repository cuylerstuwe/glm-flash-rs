//! Deployment attestation, not a self-test. An operator records completed gates
//! and pins this report to an already-tested binary/configuration artifact set.
//! Runtime health is separate and can still fail after qualification.
use anyhow::{Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{io::Read, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Qualification {
    pub report: PathBuf,
    pub sha256: String,
}
const GATES: [&str; 8] = [
    "compatibility",
    "precision_quality",
    "context_output",
    "cache_pressure",
    "native_priority",
    "fault_stability",
    "matched_performance",
    "reproducible_artifacts",
];
fn verify(bytes: &[u8], expected: &str, binary: &str) -> Result<()> {
    ensure!(
        expected.len() == 64 && format!("{:x}", Sha256::digest(bytes)) == expected,
        "qualification report differs from deployment pin"
    );
    let report: serde_json::Value = serde_json::from_slice(bytes)?;
    ensure!(
        report["version"] == 1
            && report["accepted"] == true
            && report["binary_sha256"] == binary
            && report["context_tokens"] == crate::cache_pages::CONTEXT
            && report["max_output_tokens"] == crate::cache_pages::MAX_OUTPUT,
        "qualification binary/policy/acceptance mismatch"
    );
    ensure!(
        GATES.iter().all(|g| report["gates"][g]["passed"] == true
            && report["gates"][g]["evidence"]
                .as_str()
                .is_some_and(|s| !s.is_empty())),
        "qualification has incomplete or unreferenced gates"
    );
    Ok(())
}
impl Qualification {
    fn read(&self, binary: &str) -> Result<Vec<u8>> {
        ensure!(
            self.report.is_absolute()
                && !self
                    .report
                    .components()
                    .any(|p| matches!(p, std::path::Component::ParentDir)),
            "qualification report path"
        );
        let mut bytes = Vec::new();
        std::fs::File::open(&self.report)?
            .take((1 << 20) + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 1 << 20, "qualification report size");
        verify(&bytes, &self.sha256, binary)?;
        Ok(bytes)
    }
    pub fn validate(&self, binary: &str) -> Result<()> {
        self.read(binary).map(|_| ())
    }
    pub fn deployment(&self, binary: &str, rank: usize, config: &serde_json::Value) -> Result<()> {
        let report: serde_json::Value = serde_json::from_slice(&self.read(binary)?)?;
        ensure!(
            report["deployment"][rank]["config_sha256"] == configuration_digest(config)
                && report["deployment"][rank]["environment_sha256"] == environment_digest(),
            "qualified configuration/environment differs from deployment pin"
        );
        Ok(())
    }
}
pub fn configuration_digest(config: &serde_json::Value) -> String {
    let mut value = config.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("qualification");
    }
    value.sort_all_objects();
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&value).expect("JSON value"))
    )
}
pub fn environment_digest() -> String {
    let values: std::collections::BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("ALASKA_")
                || k.starts_with("NCCL_")
                || matches!(
                    k.as_str(),
                    "LD_PRELOAD"
                        | "LD_LIBRARY_PATH"
                        | "OMP_NUM_THREADS"
                        | "CUBLAS_WORKSPACE_CONFIG"
                )
        })
        .collect();
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&values).expect("environment strings"))
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configuration_pin_ignores_only_attestation_and_key_order() {
        let a = serde_json::json!({"rank":0,"speculation":{"cost":2.14},"experimental":false});
        let mut b = serde_json::json!({"experimental":false,"speculation":{"cost":2.14},"rank":0,"qualification":{"sha256":"later"}});
        assert_eq!(configuration_digest(&a), configuration_digest(&b));
        b["speculation"]["cost"] = serde_json::json!(2.15);
        assert_ne!(configuration_digest(&a), configuration_digest(&b));
    }
    #[test]
    fn qualification_requires_all_gates_policy_and_exact_binary() {
        let mut report = serde_json::json!({"version":1,"accepted":true,"binary_sha256":"a".repeat(64),
            "context_tokens":1048576,"max_output_tokens":131072,"gates":{}});
        for gate in GATES {
            report["gates"][gate] =
                serde_json::json!({"passed":true,"evidence":"evidence/receipt.json"});
        }
        let check = |v: &serde_json::Value, binary: &str| {
            let b = serde_json::to_vec(v).unwrap();
            verify(&b, &format!("{:x}", Sha256::digest(&b)), binary)
        };
        assert!(check(&report, &"a".repeat(64)).is_ok());
        assert!(check(&report, &"b".repeat(64)).is_err());
        for gate in GATES {
            let mut bad = report.clone();
            bad["gates"][gate]["passed"] = false.into();
            assert!(check(&bad, &"a".repeat(64)).is_err());
        }
        for (key, value) in [
            ("accepted", serde_json::json!(false)),
            ("context_tokens", serde_json::json!(524288)),
            ("max_output_tokens", serde_json::json!(262144)),
        ] {
            let mut bad = report.clone();
            bad[key] = value;
            assert!(check(&bad, &"a".repeat(64)).is_err());
        }
        assert!(
            verify(
                &serde_json::to_vec(&report).unwrap(),
                &"0".repeat(64),
                &"a".repeat(64)
            )
            .is_err()
        );
    }
}
