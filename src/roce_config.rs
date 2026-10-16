//! Deployment-specific RoCE rail selection, without GPU or filesystem access.
use anyhow::{Context, Result, ensure};

pub(crate) fn rails(value: &str) -> Result<Vec<(String, i32)>> {
    let result = value
        .split(',')
        .map(|item| -> Result<(String, i32)> {
            let (name, index) = item
                .trim()
                .split_once(':')
                .context("RoCE rail must be HCA:GID")?;
            ensure!(
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
                "invalid RoCE HCA name"
            );
            ensure!(name != "." && name != "..", "invalid RoCE HCA name");
            let gid: i32 = index.parse().context("invalid RoCE GID index")?;
            ensure!((0..=255).contains(&gid), "RoCE GID index out of range");
            Ok((name.to_owned(), gid))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        result.len() == 2 && result[0].0 != result[1].0,
        "two distinct RoCE HCAs required"
    );
    Ok(result)
}
#[cfg(test)]
mod rail_tests {
    use super::*;
    #[test]
    fn explicit_rails_preserve_order_and_reject_ambiguous_or_unsafe_configuration() {
        assert_eq!(
            rails("mlx5_1:7,mlx5_0:3").unwrap(),
            vec![("mlx5_1".into(), 7), ("mlx5_0".into(), 3)]
        );
        for bad in [
            "",
            "mlx5_0:3",
            "mlx5_0:3,mlx5_0:4",
            "../x:3,y:3",
            ".:3,y:3",
            "x:-1,y:3",
            "x:256,y:3",
            "x:3,y:3,z:3",
            "x:3,y:3;cmd",
        ] {
            assert!(rails(bad).is_err(), "accepted {bad}");
        }
    }
}
