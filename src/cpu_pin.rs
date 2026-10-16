//! Host thread placement on big/small-core hosts such as GB10 (10 Cortex-X925,
//! 10 Cortex-A725). Scheduling only; replies are unchanged. ALASKA_CPU_PIN:
//! - unset or `0`: nothing pinned.
//! - `http`: rank 0's HTTP threads on the small cores, so tokenizing and
//!   streaming never share a core with the round loop or the busy-spinning
//!   RoCE proxy.
//! - `roles`: additionally the round loop on the highest-capacity core, the
//!   RoCE proxy on the next one, and helper threads on the remaining cores.
use anyhow::{Context, Result, bail, ensure};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Http,
    Roles,
}
#[derive(Debug)]
struct Plan {
    mode: Mode,
    serve: usize,
    roce: usize,
    small: Vec<usize>,
    rest: Vec<usize>,
}
static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

fn allowed() -> Result<Vec<usize>> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    ensure!(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) }
            == 0,
        "sched_getaffinity failed"
    );
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&c| unsafe { libc::CPU_ISSET(c, &set) })
        .collect())
}
fn capacity(cpu: usize) -> Result<u32> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
    Ok(std::fs::read_to_string(&path)
        .with_context(|| format!("read {path}"))?
        .trim()
        .parse()?)
}
fn plan(mode: Mode, mut cores: Vec<(usize, u32)>) -> Result<Plan> {
    let top = cores.iter().map(|c| c.1).max().context("no allowed cpus")?;
    // Best first: highest capacity, then the higher id.
    cores.sort_by_key(|&(cpu, cap)| std::cmp::Reverse((cap, cpu)));
    let small: Vec<usize> = cores
        .iter()
        .filter(|c| u64::from(c.1) * 5 < u64::from(top) * 4)
        .map(|c| c.0)
        .collect();
    ensure!(
        !small.is_empty() && cores.len() >= small.len() + 3,
        "ALASKA_CPU_PIN needs at least three big cores and one small core"
    );
    let (serve, roce) = (cores[0].0, cores[1].0);
    let rest = cores
        .iter()
        .map(|c| c.0)
        .filter(|&c| c != serve && c != roce)
        .collect();
    Ok(Plan {
        mode,
        serve,
        roce,
        small,
        rest,
    })
}
/// Validates the setting and plans placement once, before threads exist.
pub(crate) fn init() -> Result<()> {
    let mode = match std::env::var("ALASKA_CPU_PIN").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => None,
        Ok("http") => Some(Mode::Http),
        Ok("roles") => Some(Mode::Roles),
        _ => bail!("ALASKA_CPU_PIN must be 0, http or roles"),
    };
    let plan = match mode {
        None => None,
        Some(mode) => {
            let cores = allowed()?
                .into_iter()
                .map(|c| Ok((c, capacity(c)?)))
                .collect::<Result<Vec<_>>>()?;
            let plan = plan(mode, cores)?;
            eprintln!(
                "cpu placement {:?}: serve {} roce {} small {:?}",
                plan.mode, plan.serve, plan.roce, plan.small
            );
            Some(plan)
        }
    };
    ensure!(PLAN.set(plan).is_ok(), "cpu placement initialized twice");
    Ok(())
}
fn pin(cpus: &[usize]) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cpus {
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    if unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } != 0 {
        eprintln!(
            "cpu placement failed for {cpus:?}: {}",
            std::io::Error::last_os_error()
        );
    }
}
fn with(modes: &[Mode], f: impl FnOnce(&Plan)) {
    if let Some(Some(plan)) = PLAN.get()
        && modes.contains(&plan.mode)
    {
        f(plan);
    }
}
pub(crate) fn roce_cpu() -> i32 {
    let mut cpu = -1;
    with(&[Mode::Roles], |p| cpu = p.roce as i32);
    cpu
}
pub(crate) fn http_thread() {
    with(&[Mode::Http, Mode::Roles], |p| pin(&p.small));
}
pub(crate) fn serve_thread() {
    with(&[Mode::Roles], |p| pin(&[p.serve]));
}
/// Threads the round loop spawns after pinning itself would otherwise share
/// its core.
pub(crate) fn helper_thread() {
    with(&[Mode::Roles], |p| pin(&p.rest));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gb10_plan() {
        let caps = [
            718, 718, 718, 718, 718, 997, 997, 997, 997, 997, 731, 731, 731, 731, 731, 1017, 1017,
            1017, 1017, 1024,
        ];
        let p = plan(Mode::Roles, caps.iter().copied().enumerate().collect()).unwrap();
        assert_eq!((p.serve, p.roce), (19, 18));
        assert_eq!(p.small, vec![14, 13, 12, 11, 10, 4, 3, 2, 1, 0]);
        assert_eq!(p.rest.len(), 18);
        assert!(plan(Mode::Http, vec![(0, 1024), (1, 1024)]).is_err());
    }
}
