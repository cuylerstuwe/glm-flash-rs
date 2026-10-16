use super::*;
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn inventory(job: &mut CatalogJob) -> Result<Vec<Hash>> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(order) = job.inventory()? {
            return Ok(order);
        }
        if let Some(result) = job.try_finish()? {
            result?;
            anyhow::bail!("catalog finished before inventory");
        }
        ensure!(Instant::now() < deadline, "catalog inventory deadline");
        thread::yield_now();
    }
}
fn finish(job: &mut CatalogJob) -> Result<Result<Store>> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = job.try_finish()? {
            return Ok(result);
        }
        ensure!(Instant::now() < deadline, "catalog completion deadline");
        thread::yield_now();
    }
}
pub fn probe(root: &Path, rank: u32) -> Result<serde_json::Value> {
    ensure!(!root.exists(), "catalog probe requires fresh directory");
    let id = Identity::new([[1; 32], [2; 32], [3; 32], [4; 32]], [49; 32], rank, 1024)?;
    let mut job = CatalogJob::open(root.to_owned(), id.clone(), 128 << 20, None)?;
    ensure!(inventory(&mut job)?.is_empty(), "new catalog nonempty");
    for _ in 0..10000 {
        ensure!(job.try_finish()?.is_none(), "catalog bypassed intersection");
    }
    ensure!(
        Store::open(root, id.clone(), 128 << 20).is_err(),
        "catalog lock lost"
    );
    ensure!(
        job.apply(vec![[0; 32]; MAX_KEYS + 1]).is_err(),
        "oversized intersection accepted"
    );
    job.apply(vec![])?;
    ensure!(job.apply(vec![]).is_err(), "duplicate apply accepted");
    let mut store = finish(&mut job)??;
    store.save(&[7; 64], false, |_, _, b| {
        b.fill(19);
        Ok(())
    })?;
    let key = store.order()[0];
    let mut reopened = CatalogJob::open(root.to_owned(), id.clone(), 128 << 20, Some(store))?;
    ensure!(
        inventory(&mut reopened)? == [key],
        "catalog reopen inventory"
    );
    reopened.cancel();
    ensure!(
        finish(&mut reopened)?.is_err(),
        "cancelled catalog succeeded"
    );
    let mut job = CatalogJob::open(root.to_owned(), id.clone(), 128 << 20, None)?;
    ensure!(
        inventory(&mut job)? == [key],
        "cancel removed immutable prefix"
    );
    job.apply(vec![[0; 32]])?;
    ensure!(finish(&mut job)?.is_err(), "unknown keep key accepted");
    let mut job = CatalogJob::open(root.to_owned(), id.clone(), 128 << 20, None)?;
    ensure!(
        inventory(&mut job)? == [key],
        "invalid apply corrupted files"
    );
    job.apply(vec![key])?;
    let store = finish(&mut job)??;
    let mut conflict = CatalogJob::open(root.to_owned(), id.clone(), 128 << 20, None)?;
    ensure!(
        finish(&mut conflict)?.is_err(),
        "exclusive worker reopen allowed"
    );
    let mut job = CatalogJob::open(root.to_owned(), id, 128 << 20, Some(store))?;
    ensure!(inventory(&mut job)? == [key], "final inventory");
    job.apply(vec![])?;
    let store = finish(&mut job)??;
    ensure!(store.len() == 0, "empty common intersection retained entry");
    Ok(
        serde_json::json!({"rank":rank,"worker_open_and_reopen":true,"waits_for_common_intersection":true,"exclusive_catalog":true,"rejects_oversized_duplicate_unknown_apply":true,"cancel_preserves_immutable_prefix":true,"empty_intersection_removes_entries":true,"max_inventory_keys":MAX_KEYS,"channel_capacity":1,"final_digest":crate::ssd_format::hex(&store.digest())}),
    )
}
#[test]
fn lifecycle_worker_ownership_cancel_and_intersection() {
    let root = std::env::temp_dir().join(format!(
        "alaska-catalog-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    probe(&root, 0).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
