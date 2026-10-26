//! Per-function peak-memory breakdown, gated behind the `mem-profile` feature.
//!
//! The old `mem-profile` reported a single number -- the global peak heap of one
//! evaluation -- which answers "how much RAM does this cost?" but not "*which function*
//! costs it?". This module fills that gap the same way [`crate::time_profile`] fills the
//! timing one: an instrumenting global allocator plus RAII spans that attribute memory to
//! labeled scopes, printed as a table.
//!
//! Usage: put [`mem_span!`] at the top of a scope; when the guard drops, the scope's
//! memory cost is accumulated against that label.
//!
//! ```text
//! cargo bench --bench ohvpack --features mem-profile
//! ```
//!
//! # What "memory of a function" means here
//!
//! Time is additive over disjoint intervals; live heap is not. So instead of summing, each
//! span records its **peak Δ**: the maximum live-heap high-water reached anywhere inside
//! the scope (and its callees), minus the live heap at scope entry. That is "how much this
//! function needed *on top of* whatever was already resident" -- the number that tells you
//! which function drives the peak. The **abs peak** column is the same high-water without
//! subtracting the baseline, i.e. the true resident bytes while the function ran.
//!
//! # Nesting is allowed (unlike time-profile)
//!
//! Because the metric is a peak rather than a sum, nesting one span inside another does not
//! double-count: the outer span's **abs peak** includes the inner span's allocations, so the
//! two rows read as a hierarchy. There is no `other` row and no leaves-only rule.
//!
//! Note that the same is not true of **peak Δ**, which is measured against each scope's own
//! entry: a child that starts after its parent has *freed* something (a CMux tree level,
//! which drops the previous level's GLWEs as it goes) has a lower baseline and
//! can therefore show a larger Δ than the parent containing it. Compare Δ within a level,
//! and read abs peak down the hierarchy.
//!
//! # Report layout
//!
//! Rows are printed in the order the scopes were *entered* and indented by nesting depth,
//! so the table reads as the evaluation's own structure. The benchmarks open three kinds of
//! top-level section, in this order:
//!
//!  - `lut` -- the plaintext table alone, built before anything else;
//!  - `setup` -- module, scratch, keys and selectors, on top of the resident table;
//!  - the evaluation, with one nested row per layer of the tree.
//!
//! There is deliberately no call count: the metric is a peak, so the number of times a
//! label was entered says nothing about the memory it costs.
//!
//! # Caveats
//!
//! - Allocation is tracked by `Layout::size()` at the allocator boundary; it counts the
//!   requested heap bytes, not allocator slack or memory-mapped pages.
//! - The global peak (and Δ) see allocations on every thread, but per-span attribution
//!   only follows the thread that entered the span. The heavy allocations here (the
//!   `Vec<GLWE>`/`Vec<LWE>` groups and `glwe_alloc`/`lwe_alloc`) all happen on the calling
//!   thread; per-op scratch is preallocated and reused, so it does not show up per span.

/// Accumulates the peak live-heap cost of the enclosing scope against `$label`.
///
/// Expands to nothing without the `mem-profile` feature.
#[macro_export]
macro_rules! mem_span {
    ($label:expr) => {
        let _mem_span_guard = $crate::mem_profile::Span::new($label);
    };
}

#[cfg(feature = "mem-profile")]
mod imp {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Live heap bytes (allocated minus freed), across every thread.
    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    /// High-water mark of `CURRENT` since the last [`reset`].
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    /// Monotonic counter stamped on every span entry, so the report can be printed in
    /// the order the scopes were *entered* rather than the order they completed (inner
    /// scopes always complete first, which would invert the hierarchy).
    static SEQ: AtomicUsize = AtomicUsize::new(0);

    /// A span currently on the stack: its label, the live heap at entry, the highest
    /// `CURRENT` seen while it (or a child) was active, its nesting depth and its entry
    /// order.
    struct Frame {
        label: &'static str,
        start: usize,
        peak: usize,
        depth: usize,
        seq: usize,
    }

    /// One row of the report: worst-case cost seen across every call of a label, plus the
    /// depth and entry order of its first occurrence (used only for laying the table out).
    struct Row {
        label: &'static str,
        max_delta: usize,
        max_peak: usize,
        depth: usize,
        seq: usize,
    }

    thread_local! {
        /// Active spans on this thread, innermost last.
        static SPAN_STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
        /// Accumulated per-label rows. Linear scan: only a handful of labels.
        static TABLE: RefCell<Vec<Row>> = const { RefCell::new(Vec::new()) };
    }

    /// Records `size` new live bytes and bumps the global and innermost-span peaks.
    ///
    /// Allocation-free and re-entrancy-safe: it only touches atomics and, if it can take
    /// the borrow without conflict, the top stack frame. When `SPAN_STACK` is already
    /// borrowed (i.e. we are mid-`push`/`pop` inside [`Span`]) it simply skips the frame
    /// update -- those transient bookkeeping bytes are not worth attributing.
    #[inline]
    fn on_alloc(size: usize) {
        let cur: usize = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
        PEAK.fetch_max(cur, Ordering::Relaxed);
        let _ = SPAN_STACK.try_with(|c| {
            if let Ok(mut v) = c.try_borrow_mut()
                && let Some(f) = v.last_mut()
                && cur > f.peak
            {
                f.peak = cur;
            }
        });
    }

    #[inline]
    fn on_dealloc(size: usize) {
        CURRENT.fetch_sub(size, Ordering::Relaxed);
    }

    /// Global allocator that forwards to [`System`] and tracks live heap. Install it in a
    /// benchmark with `#[global_allocator] static A: TrackingAlloc = TrackingAlloc;`.
    pub struct TrackingAlloc;

    unsafe impl GlobalAlloc for TrackingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwarding the caller's own contract to the System allocator.
            let p: *mut u8 = unsafe { System.alloc(layout) };
            if !p.is_null() {
                on_alloc(layout.size());
            }
            p
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: `ptr`/`layout` come from a prior `alloc` on this same allocator.
            unsafe { System.dealloc(ptr, layout) };
            on_dealloc(layout.size());
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwarding the caller's own contract to the System allocator.
            let p: *mut u8 = unsafe { System.alloc_zeroed(layout) };
            if !p.is_null() {
                on_alloc(layout.size());
            }
            p
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: `ptr`/`layout` come from a prior `alloc`; `new_size` upholds the
            // `GlobalAlloc::realloc` contract as guaranteed by the caller.
            let p: *mut u8 = unsafe { System.realloc(ptr, layout, new_size) };
            if !p.is_null() {
                if new_size >= layout.size() {
                    on_alloc(new_size - layout.size());
                } else {
                    on_dealloc(layout.size() - new_size);
                }
            }
            p
        }
    }

    /// RAII guard: snapshots live heap on creation and records the scope's cost on drop.
    pub struct Span;

    impl Span {
        pub fn new(label: &'static str) -> Self {
            let cur: usize = CURRENT.load(Ordering::Relaxed);
            let seq: usize = SEQ.fetch_add(1, Ordering::Relaxed);
            SPAN_STACK.with(|c| {
                let mut v = c.borrow_mut();
                let depth: usize = v.len();
                v.push(Frame { label, start: cur, peak: cur, depth, seq });
            });
            Span
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let frame: Frame = match SPAN_STACK.with(|c| c.borrow_mut().pop()) {
                Some(f) => f,
                None => return,
            };
            // Propagate this scope's high-water to its parent so the parent's peak Δ
            // includes the memory its children held.
            SPAN_STACK.with(|c| {
                if let Some(parent) = c.borrow_mut().last_mut()
                    && frame.peak > parent.peak
                {
                    parent.peak = frame.peak;
                }
            });
            let delta: usize = frame.peak.saturating_sub(frame.start);
            TABLE.with(|c| {
                let mut t = c.borrow_mut();
                if let Some(r) = t.iter_mut().find(|r| r.label == frame.label) {
                    r.max_delta = r.max_delta.max(delta);
                    r.max_peak = r.max_peak.max(frame.peak);
                    // A label reached from several call sites is laid out at its shallowest,
                    // earliest occurrence.
                    r.depth = r.depth.min(frame.depth);
                    r.seq = r.seq.min(frame.seq);
                } else {
                    t.push(Row {
                        label: frame.label,
                        max_delta: delta,
                        max_peak: frame.peak,
                        depth: frame.depth,
                        seq: frame.seq,
                    });
                }
            });
        }
    }

    /// Clears the per-label table and re-bases the peak to the currently live heap.
    ///
    /// Call it right before the run you want to measure (typically after setup): the peak
    /// and every span's baseline then start from the resident setup memory, so the report
    /// attributes only what the evaluation itself adds.
    pub fn reset() {
        TABLE.with(|c| c.borrow_mut().clear());
        SPAN_STACK.with(|c| c.borrow_mut().clear());
        SEQ.store(0, Ordering::Relaxed);
        PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Global peak live heap since the last [`reset`], in bytes.
    pub fn global_peak() -> usize {
        PEAK.load(Ordering::Relaxed)
    }

    /// Prints the breakdown in entry order, indented by nesting depth.
    ///
    /// Entry order, not descending cost: the sections a caller wants to compare -- the
    /// table, the setup, then one row per tree layer -- are laid out top to bottom the way
    /// the evaluation walks them, and a nested row sits under the row that contains it.
    pub fn report(header: &str) {
        let mut rows: Vec<(&'static str, usize, usize, usize, usize)> = TABLE.with(|c| {
            c.borrow()
                .iter()
                .map(|r| (r.label, r.max_delta, r.max_peak, r.depth, r.seq))
                .collect()
        });
        rows.sort_by_key(|r| r.4);

        let peak: usize = PEAK.load(Ordering::Relaxed);
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);

        println!("\n[mem-profile] {header}  global peak = {:.2} MiB ({peak} bytes)", mib(peak));
        println!("  {:<46} {:>12} {:>12}", "section", "peak Δ", "abs peak");
        for (label, delta, abs, depth, _) in &rows {
            let indent: usize = depth * 2;
            println!(
                "  {:indent$}{:<width$} {:>9.2}MiB {:>9.2}MiB",
                "",
                label,
                mib(*delta),
                mib(*abs),
                indent = indent,
                width = 46usize.saturating_sub(indent),
            );
        }
    }

    /// Static label for the `n`-th layer of a tree, so a loop over layers can open a span
    /// per iteration ([`Span`] needs a `&'static str`, which rules out formatting one).
    ///
    /// Layers past the table fall back to a shared bucket rather than panicking; no
    /// construction here goes near that depth.
    pub fn layer_label(n: usize) -> &'static str {
        const LABELS: [&str; 17] = [
            "layer 0", "layer 1", "layer 2", "layer 3", "layer 4", "layer 5", "layer 6",
            "layer 7", "layer 8", "layer 9", "layer 10", "layer 11", "layer 12", "layer 13",
            "layer 14", "layer 15", "layer 16+",
        ];
        LABELS[n.min(LABELS.len() - 1)]
    }
}

#[cfg(not(feature = "mem-profile"))]
mod imp {
    /// No-op stand-in; the guard holds nothing and `Drop` is never generated.
    pub struct Span;

    impl Span {
        #[inline(always)]
        pub fn new(_label: &'static str) -> Self {
            Span
        }
    }

    #[inline(always)]
    pub fn reset() {}

    #[inline(always)]
    pub fn global_peak() -> usize {
        0
    }

    #[inline(always)]
    pub fn report(_header: &str) {}

    #[inline(always)]
    pub fn layer_label(_n: usize) -> &'static str {
        ""
    }
}

#[cfg(feature = "mem-profile")]
pub use imp::TrackingAlloc;
pub use imp::{Span, global_peak, layer_label, report, reset};
