//! Per-operation timing breakdown, gated behind the `time-profile` feature.
//!
//! Criterion measures whole closures and cannot attribute time to the subfunctions inside
//! one, so it cannot answer "how much of an evaluation is external products?". This module
//! fills that gap the same way `mem-profile` fills the memory one: a cargo feature swaps
//! the benchmark from Criterion to a single instrumented run that prints a table.
//!
//! Usage: put [`time_span!`] at the top of a scope and the elapsed time is accumulated
//! against that label when the guard drops.
//!
//! ```text
//! cargo bench --bench ohvpack --features time-profile
//! ```
//!
//! # Only instrument leaves
//!
//! [`report`] treats every span as a leaf and derives an `other` row as
//! `total - sum(spans)`. Nesting one instrumented span inside another therefore
//! double-counts and makes `other` negative (reported as `0`). Span the innermost
//! operations only -- e.g. `cmux_assign`, not the tree level that loops over it.
//!
//! # Reading the numbers
//!
//! Each span costs two `Instant::now()` calls (~25ns), so the overhead scales with the
//! number of spans entered, not with the work done. At ohvpack N=1024, d=2 that is ~3k spans,
//! about 0.15ms against an 11ms evaluation -- about a percent, and the measured total
//! tracks Criterion's closely. It only becomes worth thinking about for labels entered
//! hundreds of thousands of times.
//!
//! The total here is one cold run, whereas Criterion reports a converged average over
//! many, so the two differ by a few percent for reasons unrelated to instrumentation.
//! Trust this table for *shares*, and Criterion for absolute wall-clock.

/// Accumulates the elapsed time of the enclosing scope against `$label`.
///
/// Expands to nothing without the `time-profile` feature.
#[macro_export]
macro_rules! time_span {
    ($label:expr) => {
        let _time_span_guard = $crate::time_profile::Span::new($label);
    };
}

#[cfg(feature = "time-profile")]
mod imp {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    thread_local! {
        /// (label, total, calls). Linear scan: a handful of labels, so a map would cost
        /// more than it saves on a path this hot.
        static SPANS: RefCell<Vec<(&'static str, Duration, u64)>> = const { RefCell::new(Vec::new()) };
    }

    /// RAII guard recording into the thread-local table on drop.
    pub struct Span {
        label: &'static str,
        start: Instant,
    }

    impl Span {
        pub fn new(label: &'static str) -> Self {
            Self { label, start: Instant::now() }
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            SPANS.with(|s| {
                let mut s = s.borrow_mut();
                if let Some(e) = s.iter_mut().find(|e| e.0 == self.label) {
                    e.1 += elapsed;
                    e.2 += 1;
                } else {
                    s.push((self.label, elapsed, 1));
                }
            });
        }
    }

    /// Drops every accumulated span. Call before the run you want to measure.
    pub fn reset() {
        SPANS.with(|s| s.borrow_mut().clear());
    }

    /// Prints the breakdown, sorted by descending time, with each label's share of
    /// `total` and an `other` row for the time outside every span.
    pub fn report(header: &str, total: Duration) {
        let mut rows: Vec<(&'static str, Duration, u64)> = SPANS.with(|s| s.borrow().clone());
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));

        let spanned: Duration = rows.iter().map(|r| r.1).sum();
        let total_ms = total.as_secs_f64() * 1e3;
        let pct = |d: Duration| if total_ms > 0.0 { d.as_secs_f64() * 1e3 / total_ms * 100.0 } else { 0.0 };

        println!("\n[time-profile] {header}  total = {total_ms:.2} ms");
        println!("  {:<28} {:>10} {:>8} {:>10}", "operation", "time", "share", "calls");
        for (label, d, calls) in &rows {
            println!("  {:<28} {:>8.2}ms {:>7.1}% {:>10}", label, d.as_secs_f64() * 1e3, pct(*d), calls);
        }
        // Saturating: nested spans would double-count and push this negative.
        let other: Duration = total.saturating_sub(spanned);
        println!("  {:<28} {:>8.2}ms {:>7.1}% {:>10}", "other", other.as_secs_f64() * 1e3, pct(other), "-");
        if spanned > total {
            println!("  WARNING: spans sum to more than the total -- nested spans double-count.");
        }
    }
}

#[cfg(not(feature = "time-profile"))]
mod imp {
    use std::time::Duration;

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
    pub fn report(_header: &str, _total: Duration) {}
}

pub use imp::{Span, report, reset};
