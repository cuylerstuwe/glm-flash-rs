//! Bounded CPU image preparation retained until the actual worker exits.
//! The serving adapter owns admission, cancellation and worker retirement.
use crate::{
    vision_codecs::Codecs,
    vision_decode,
    vision_fetch::Client,
    vision_gif,
    vision_input::{ImagePrompt, ImageRows, Registry},
    vision_markup::Reference,
    vision_preprocess::{Plan, Prepared, Settings},
    vision_request::{ActiveDeadline, Control, Lease},
    vision_source,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub encoded: usize,
    pub scratch: usize,
    pub retained: usize,
}
impl Limits {
    pub(crate) fn total(self) -> Result<usize> {
        self.encoded
            .checked_add(self.scratch)
            .and_then(|v| v.checked_add(self.retained))
            .context("image reservation overflow")
    }
}
pub(crate) struct Image {
    pub encoded_sha256: [u8; 32],
    pub max_tokens: usize,
    pub prepared: Prepared,
}
pub(crate) struct Encoded {
    bytes: Vec<u8>,
    max_tokens: usize,
}
impl Encoded {
    pub(crate) fn new(bytes: Vec<u8>, max_tokens: usize) -> Result<Self> {
        ensure!(
            !bytes.is_empty() && bytes.capacity() <= vision_decode::MAX_ENCODED_BYTES,
            "image encoded byte bound"
        );
        ensure!((16..=8000).contains(&max_tokens), "image token bound");
        Ok(Self { bytes, max_tokens })
    }
}
enum Input {
    Encoded(Encoded),
    Reference {
        source: Reference,
        max_tokens: usize,
    },
}
/// Ordered references from the private canonical renderer. No bytes are loaded
/// until a native ticket has reserved `encoded_charge` plus workspace/canvases.
pub(crate) struct Sources {
    inputs: Vec<Input>,
    encoded_charge: usize,
    client: Option<Client>,
}
impl Sources {
    pub(crate) fn new(
        references: Vec<Reference>,
        cap: usize,
        low_cap: usize,
        fetch: bool,
    ) -> Result<Self> {
        ensure!(
            !references.is_empty() && references.len() <= 8,
            "image count bound"
        );
        let source_bytes = references.iter().try_fold(0usize, |n, r| {
            n.checked_add(r.url.capacity())
                .context("image source capacity overflow")
        })?;
        ensure!(
            source_bytes <= crate::chat_json::BODY_BYTES,
            "image source capacity bound"
        );
        // All source strings remain owned; one source is materialized at a
        // time. Base64 normalization may temporarily duplicate the largest
        // source, alongside its decoded/downloaded byte buffer.
        let largest = references.iter().map(|r| r.url.len()).max().unwrap_or(0);
        let encoded_charge = source_bytes
            .checked_add(largest)
            .and_then(|n| n.checked_add(vision_decode::MAX_ENCODED_BYTES))
            .context("image source reservation overflow")?;
        Ok(Self {
            inputs: references
                .into_iter()
                .map(|source| {
                    let max_tokens = source.token_budget(cap, low_cap);
                    Input::Reference { source, max_tokens }
                })
                .collect(),
            encoded_charge,
            client: fetch.then(Client::new),
        })
    }
    pub(crate) fn encoded_charge(&self) -> usize {
        self.encoded_charge
    }
}
pub(crate) struct Ready {
    pub images: Vec<Image>,
    lease: Lease,
}
impl Ready {
    pub(crate) fn ticket(&self) -> u64 {
        self.lease.ticket()
    }
    /// Only decoded, canonically prepared images may introduce virtual rows.
    /// The resulting owner keeps canvases and reservation together until all
    /// image/model work retires; the existing text-only queue cannot accept it.
    pub(crate) fn bind(
        self,
        registry: &mut Registry,
        text: &[i32],
        output: usize,
    ) -> Result<Bound> {
        ensure!(
            (1..=crate::cache_pages::MAX_OUTPUT).contains(&output),
            "image output policy"
        );
        let rows = self
            .images
            .iter()
            .map(|image| registry.register(image.prepared.key(), image.prepared.plan().tokens()))
            .collect::<Result<Vec<_>>>()?;
        let prompt = ImagePrompt::expand(text, &rows)?;
        ensure!(
            !prompt.tokens().is_empty()
                && prompt.tokens().len() <= crate::cache_pages::CONTEXT - output,
            "expanded image prompt plus output exceeds context"
        );
        Ok(Bound {
            prompt,
            images: self
                .images
                .into_iter()
                .zip(rows)
                .map(|(image, rows)| BoundImage { image, rows })
                .collect(),
            lease: self.lease,
        })
    }
}
pub(crate) struct BoundImage {
    pub image: Image,
    pub rows: ImageRows,
}
pub(crate) struct Bound {
    pub prompt: ImagePrompt,
    pub images: Vec<BoundImage>,
    lease: Lease,
}
impl Bound {
    pub(crate) fn ticket(&self) -> u64 {
        self.lease.ticket()
    }
}
pub(crate) enum Poll {
    Pending,
    Held,
    Ready(Ready),
}
pub(crate) struct Job {
    control: Arc<Control>,
    thread: Option<JoinHandle<Result<Vec<Image>>>>,
    finished: Option<Result<Vec<Image>>>,
    lease: Option<Lease>,
}
impl Job {
    /// The native owner reserves the complete ticket budget before materializing
    /// inputs; this method validates that retained inputs fit that reservation.
    pub(crate) fn spawn(
        lease: Lease,
        inputs: Vec<Encoded>,
        settings: Arc<Settings>,
        native: Option<Arc<Codecs>>,
        limits: Limits,
    ) -> Result<Self> {
        let charge = inputs.iter().try_fold(0usize, |n, v| {
            n.checked_add(v.bytes.capacity())
                .context("image input capacity overflow")
        })?;
        Self::spawn_inner(
            lease,
            Sources {
                inputs: inputs.into_iter().map(Input::Encoded).collect(),
                encoded_charge: charge,
                client: None,
            },
            settings,
            native,
            limits,
        )
    }
    pub(crate) fn spawn_sources(
        lease: Lease,
        sources: Sources,
        settings: Arc<Settings>,
        native: Option<Arc<Codecs>>,
        limits: Limits,
    ) -> Result<Self> {
        Self::spawn_inner(lease, sources, settings, native, limits)
    }
    fn spawn_inner(
        lease: Lease,
        sources: Sources,
        settings: Arc<Settings>,
        native: Option<Arc<Codecs>>,
        limits: Limits,
    ) -> Result<Self> {
        let Sources {
            inputs,
            encoded_charge,
            client,
        } = sources;
        ensure!(!inputs.is_empty() && inputs.len() <= 8, "image count bound");
        ensure!(
            encoded_charge <= limits.encoded && limits.total()? <= lease.bytes(),
            "image reservation too small"
        );
        let control = Control::new();
        let worker = control.clone();
        let thread = thread::Builder::new()
            .name("alaska-image-prepare".into())
            .spawn(move || {
                let mut images = Vec::with_capacity(inputs.len());
                let mut retained = 0usize;
                let deadline = ActiveDeadline::new(&worker, Duration::from_secs(30))?;
                for input in inputs {
                    worker.check()?;
                    let Encoded { bytes, max_tokens } = match input {
                        Input::Encoded(encoded) => encoded,
                        Input::Reference { source, max_tokens } => {
                            let bytes = if source.url.starts_with("data:") {
                                vision_source::inline(
                                    &source.url,
                                    vision_decode::MAX_ENCODED_BYTES,
                                )?
                            } else {
                                client
                                    .as_ref()
                                    .context("image URL fetching is disabled")?
                                    .fetch(
                                        &source.url,
                                        vision_decode::MAX_ENCODED_BYTES,
                                        deadline.deadline(&worker)?,
                                        &worker,
                                    )?
                            };
                            // Retain the exact fetched result if foreground work
                            // arrived during DNS/HTTP. Never cancel/retry on hold.
                            worker.check()?;
                            Encoded::new(bytes, max_tokens)?
                        }
                    };
                    let encoded_sha256 = Sha256::digest(&bytes).into();
                    let decoded = if bytes.starts_with(b"\x89PNG") {
                        vision_decode::png(&bytes, limits.scratch, || worker.check())?
                    } else if bytes.starts_with(b"GIF") {
                        vision_gif::decode(&bytes, limits.scratch, || worker.check())?
                    } else if bytes.starts_with(&[0xff, 0xd8]) {
                        native
                            .as_ref()
                            .context("native JPEG decoder unavailable")?
                            .jpeg(&bytes, limits.scratch, || worker.check())?
                    } else {
                        native
                            .as_ref()
                            .context("native WebP decoder unavailable")?
                            .webp(&bytes, limits.scratch, || worker.check())?
                    };
                    let plan = Plan::new(decoded.height, decoded.width, max_tokens)?;
                    let (ch, cw) = plan.content_size();
                    let (ph, pw) = plan.canvas_size();
                    // Reserve both sides of each separable resize pass plus bounded
                    // axis metadata. Earlier prepared canvases use retained credit.
                    let intermediate = decoded
                        .height
                        .checked_mul(cw)
                        .and_then(|v| v.checked_mul(3))
                        .context("image resize size")?;
                    let resized = ch * cw * 3;
                    let canvas = ph * pw * 3;
                    let horizontal = if decoded.width == cw {
                        decoded.rgb.len()
                    } else {
                        decoded.rgb.len() + intermediate
                    };
                    let vertical = if decoded.height == ch {
                        intermediate
                    } else {
                        intermediate + resized
                    };
                    let peak = horizontal
                        .max(vertical)
                        .max(resized + canvas)
                        .checked_add(32 * 1024 * 1024)
                        .context("image resize workspace")?;
                    ensure!(
                        peak <= limits.scratch,
                        "image preparation scratch admission"
                    );
                    retained = retained
                        .checked_add(canvas)
                        .context("image retained overflow")?;
                    ensure!(
                        retained <= limits.retained,
                        "image prepared retention admission"
                    );
                    worker.check()?;
                    let prepared =
                        Prepared::from_rgb_checked(plan, decoded.rgb, &settings, || {
                            worker.check()
                        })?;
                    images.push(Image {
                        encoded_sha256,
                        max_tokens,
                        prepared,
                    });
                }
                worker.check()?;
                Ok(images)
            })
            .context("spawn bounded image worker")?;
        Ok(Self {
            control,
            thread: Some(thread),
            finished: None,
            lease: Some(lease),
        })
    }
    pub(crate) fn hold(&self, held: bool) -> Result<()> {
        self.control.hold(held)
    }
    pub(crate) fn cancel(&self) {
        self.control.cancel();
    }
    pub(crate) fn poll(&mut self) -> Result<Poll> {
        ensure!(self.lease.is_some(), "image job already retired");
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let result = self
                .thread
                .take()
                .expect("finished image thread")
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("image worker panicked")));
            self.finished = Some(result);
        }
        let (held, waiting, cancelled, _) = self.control.status()?;
        if self.finished.is_none() {
            return Ok(if held && waiting {
                Poll::Held
            } else {
                Poll::Pending
            });
        }
        if held {
            return Ok(Poll::Held);
        }
        let result = self.finished.take().expect("joined image work");
        let lease = self.lease.take().expect("image reservation");
        if cancelled {
            // Release potentially completed canvases before returning admission
            // credit to a concurrent ticket owner.
            drop(result);
            drop(lease);
            anyhow::bail!("image preparation cancelled");
        }
        Ok(Poll::Ready(Ready {
            images: result?,
            lease,
        }))
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        self.control.cancel();
        if let Some(thread) = self.thread.take() {
            if thread.is_finished() {
                let _ = thread.join();
            } else {
                drop(thread);
                if let Some(lease) = self.lease.take() {
                    lease.abandon();
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "vision_source_job_tests.rs"]
mod source_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vision_fixtures, vision_request::Budget};
    const PNG: &[u8] = include_bytes!("../native/vision-decode/fixtures/color2-depth8-plain.png");
    pub(super) fn settings() -> Arc<Settings> {
        Arc::new(
            Settings::canonical(
                &serde_json::from_str(vision_fixtures::CONFIG).unwrap(),
                &serde_json::from_str(vision_fixtures::PROCESSOR).unwrap(),
            )
            .unwrap(),
        )
    }
    fn make() -> (Arc<Budget>, Job) {
        let limits = Limits {
            encoded: PNG.len() * 2,
            scratch: 384 * 1024 * 1024,
            retained: 2 * 1024 * 1024,
        };
        let budget = Budget::new(limits.total().unwrap()).unwrap();
        let lease = budget.reserve(17, limits.total().unwrap()).unwrap();
        let job = Job::spawn(
            lease,
            vec![
                Encoded::new(PNG.to_vec(), 8000).unwrap(),
                Encoded::new(PNG.to_vec(), 8000).unwrap(),
            ],
            settings(),
            None,
            limits,
        )
        .unwrap();
        (budget, job)
    }
    pub(super) fn poll_until(job: &mut Job, held: bool) -> Result<Option<Ready>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            match job.poll()? {
                Poll::Ready(v) => return Ok(Some(v)),
                Poll::Held if held => return Ok(None),
                _ => {}
            }
            ensure!(
                std::time::Instant::now() < deadline,
                "image worker test deadline"
            );
            thread::yield_now();
        }
    }
    #[test]
    fn real_decode_hold_retains_ticket_and_exact_prepared_images() {
        let (budget, mut job) = make();
        let reserved = budget.used().unwrap();
        job.hold(true).unwrap();
        assert!(poll_until(&mut job, true).unwrap().is_none());
        assert_eq!(budget.used().unwrap(), reserved);
        job.hold(false).unwrap();
        let ready = poll_until(&mut job, false).unwrap().unwrap();
        assert_eq!(ready.ticket(), 17);
        assert_eq!(ready.images.len(), 2);
        assert_eq!(
            ready.images[0].prepared.key(),
            ready.images[1].prepared.key()
        );
        assert_eq!(
            ready.images[0].encoded_sha256,
            <[u8; 32]>::from(Sha256::digest(PNG))
        );
        let decoded = vision_decode::png(PNG, 384 * 1024 * 1024, || Ok(())).unwrap();
        let cold = Prepared::from_rgb(
            Plan::new(decoded.height, decoded.width, 8000).unwrap(),
            decoded.rgb,
            &settings(),
        )
        .unwrap();
        assert_eq!(ready.images[0].prepared.key(), cold.key());
        assert_eq!(budget.used().unwrap(), reserved);
        drop(ready);
        drop(job);
        assert_eq!(budget.used().unwrap(), 0);
    }
    #[test]
    fn held_cancel_retires_before_reservation_is_reused() {
        let (budget, mut job) = make();
        job.hold(true).unwrap();
        poll_until(&mut job, true).unwrap();
        job.cancel();
        assert!(poll_until(&mut job, false).is_err());
        drop(job);
        assert_eq!(budget.used().unwrap(), 0);
        assert!(budget.reserve(18, 1).is_ok());
    }
    #[test]
    fn each_image_retains_its_budget_and_binds_only_validated_rows() {
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 448, 448);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&vec![127; 448 * 448 * 3])
                .unwrap();
        }
        let limits = Limits {
            encoded: png.len() + png.capacity(),
            scratch: 384 << 20,
            retained: 2 << 20,
        };
        let budget = Budget::new(limits.total().unwrap()).unwrap();
        let lease = budget.reserve(21, limits.total().unwrap()).unwrap();
        let mut job = Job::spawn(
            lease,
            vec![
                Encoded::new(png.clone(), 16).unwrap(),
                Encoded::new(png, 64).unwrap(),
            ],
            settings(),
            None,
            limits,
        )
        .unwrap();
        let ready = poll_until(&mut job, false).unwrap().unwrap();
        assert_eq!(
            ready
                .images
                .iter()
                .map(|i| i.max_tokens)
                .collect::<Vec<_>>(),
            [16, 64]
        );
        assert_eq!(
            ready
                .images
                .iter()
                .map(|i| i.prepared.plan().tokens())
                .collect::<Vec<_>>(),
            [16, 64]
        );
        assert_ne!(
            ready.images[0].prepared.key(),
            ready.images[1].prepared.key()
        );
        let mut registry = Registry::new(262144).unwrap();
        let bound = ready
            .bind(
                &mut registry,
                &[42, 154830, 154854, 154831, 154830, 154854, 154831],
                128,
            )
            .unwrap();
        assert_eq!(bound.ticket(), 21);
        assert_eq!(bound.prompt.tokens().len(), 85);
        for image in &bound.images {
            assert_eq!(image.rows.key(), image.image.prepared.key());
            for (row, &id) in image.rows.ids().iter().enumerate() {
                assert_eq!(bound.prompt.row(id), Some((image.rows.key(), row as u32)));
            }
        }
        assert!(budget.used().unwrap() > 0);
        drop(bound);
        assert_eq!(budget.used().unwrap(), 0);
    }
    #[test]
    fn context_expansion_failure_retires_the_original_reservation() {
        let (budget, mut job) = make();
        let ready = poll_until(&mut job, false).unwrap().unwrap();
        let mut registry = Registry::new(262144).unwrap();
        let output = crate::cache_pages::MAX_OUTPUT;
        let mut text = vec![42; crate::cache_pages::CONTEXT - output];
        text[..6].copy_from_slice(&[154830, 154854, 154831, 154830, 154854, 154831]);
        assert!(ready.bind(&mut registry, &text, output).is_err());
        assert_eq!(budget.used().unwrap(), 0);
        assert!(Encoded::new(PNG.to_vec(), 15).is_err());
    }
}
