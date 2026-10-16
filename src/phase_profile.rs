//! Optional diagnostic GPU/host phase tracing. Events are allocated before work;
//! no event waits/readbacks occur until the complete profiled slab has drained.
use crate::cuda::{Device, Event, Stream};
use anyhow::{Result, ensure};
use std::{rc::Rc, time::Instant};
pub struct Trace {
    events: Vec<Event>,
    marks: Vec<(&'static str, usize, Instant)>,
}
impl Trace {
    pub fn new(d: &Rc<Device>) -> Result<Self> {
        Ok(Self {
            events: (0..400).map(|_| d.event()).collect::<Result<_>>()?,
            marks: Vec::with_capacity(400),
        })
    }
    pub fn mark(&mut self, s: &Stream, label: &'static str, layer: usize) -> Result<()> {
        ensure!(self.marks.len() < self.events.len(), "phase trace capacity");
        self.events[self.marks.len()].record(s)?;
        self.marks.push((label, layer, Instant::now()));
        Ok(())
    }
    pub fn finish(self) -> Result<serde_json::Value> {
        ensure!(self.marks.len() > 1, "empty phase trace");
        let mut phases = Vec::new();
        for i in 0..self.marks.len() - 1 {
            let (label, layer, start) = self.marks[i];
            phases.push(serde_json::json!({"phase":label,"layer":layer,"gpu_ms":self.events[i+1].elapsed_after(&self.events[i])?,"host_enqueue_ms":self.marks[i+1].2.duration_since(start).as_secs_f64()*1000.}));
        }
        Ok(
            serde_json::json!({"phases":phases,"gpu_elapsed_ms":self.events[self.marks.len()-1].elapsed_after(&self.events[0])?,"scope":"diagnostic CUDA stream elapsed time (includes idle/collective waits); host enqueue includes driver stalls, not CPU utilization; no per-phase synchronization"}),
        )
    }
}
