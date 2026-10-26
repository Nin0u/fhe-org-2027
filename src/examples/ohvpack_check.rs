//! Correctness of OHVPack and VPack at a real parameter set (not the toy one).
//!
//! `PARAMS_FILE=... cargo run --release --example ohvpack_check`
//!
//! The integration tests run against `init_toy()`, whose torus precision is deliberately
//! generous, so they say nothing about whether a construction survives the parameter sets
//! `paramux` actually emits. This does: it evaluates the same scrambled table at the same
//! indices under both and reports how many decrypt correctly.
use ohvpack::context::{MuxLayouts, SecretKeys, scratch_bytes, secret_keys};
use ohvpack::lut::{Lut, build_lut};
use ohvpack::ohvpack::ohvpack_eval;
use ohvpack::params::params_from_json;
use ohvpack::prepare_data::prepare_glwe;
use ohvpack::selector::{ohvpack_make_selectors, vpack_make_selectors_tmp_bytes};
use ohvpack::vertical_packing::vpack_eval_tmp_bytes;
use poulpy_core::layouts::{BackendLWEPlaintext, ModuleCoreAlloc};
use poulpy_core::{LWESampleExtract, LWEDecrypt};
use poulpy_hal::{api::{ScratchOwnedAlloc, ScratchOwnedBorrow}, layouts::{Backend, CnvPVecL, Module, ScratchOwned}};
use poulpy_cpu_ref::FFT64Ref as B;

fn main() {
    let path = std::env::var("PARAMS_FILE").expect("PARAMS_FILE");
    for mut params in params_from_json(&path) {
        let layouts = MuxLayouts::new(&params);
        let module: Module<B> = Module::<B>::new(params.n_rlwe as u64);
        let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(
            scratch_bytes(&module, &layouts)
                .max(vpack_make_selectors_tmp_bytes(&module, params.n_rlwe, &layouts.ggsw))
                .max(vpack_eval_tmp_bytes::<B, _, _>(&module, &layouts.glwe, &layouts.ggsw)),
        );
        let sk: SecretKeys<B> = secret_keys(&module, &mut params, &layouts);
        let lut: Vec<i32> = build_lut(params.n_rlwe.pow(params.depth as u32), params.p);

        let p = params.p as i64;
        let idxs: Vec<u64> = vec![0, 2048, 12345, (lut.len() - 1) as u64];
        let mut ok_ohvpack = 0usize;
        for &k in &idxs {
            let want = (lut[k as usize] as i64).rem_euclid(p);

            // --- OHVPack ---
            let (gl2, ts) = ohvpack_make_selectors::<B>(&module, &mut params, &layouts.glwe,
                &layouts.ggsw, &sk.glwe_dft, k as i64, &mut scratch.borrow());
            let glp2: Vec<CnvPVecL<Vec<u8>, <B as Backend>::DftWord, B>> =
                gl2.iter().map(|s| prepare_glwe(&module, s, &mut scratch.borrow())).collect();
            let r2 = ohvpack_eval::<B>(&module, &params, &mut scratch.borrow(),
                Lut::Streaming(&lut), &glp2, &ts);
            let mut pt2: BackendLWEPlaintext<B> = module.lwe_plaintext_alloc(params.base2k.into(), params.k_rlwe.into());
            module.lwe_decrypt(&r2, &mut pt2, &sk.lwe, &mut scratch.borrow());
            let got2 = pt2.decode_i64(params.log_p.into()).rem_euclid(p);
            if got2 == want { ok_ohvpack += 1; } else { println!("  ohvpack k={k}: got {got2} want {want}"); }
        }
        // --- VPack, on the same table and indices ---
        let mut ok_vpack = 0usize;
        {
            use ohvpack::selector::vpack_make_selectors;
            use ohvpack::trivial::trivial_glwe;
            use ohvpack::vertical_packing::{FirstLevel, vpack_eval_streaming};
            use poulpy_core::layouts::{BackendGLWE, BackendGLWEPlaintext, BackendLWE};
            let n = params.n_rlwe;
            let log_n = n.trailing_zeros() as usize;
            let log_table = log_n * params.depth;
            let n_cand = 1usize << (log_table - log_n);
            for &k in &idxs {
                let (tree_sel, rot_sel) = vpack_make_selectors::<u64, B>(&module, &mut params,
                    &layouts.ggsw, &sk.glwe_dft, k, &mut scratch.borrow());
                let mut pt_c: BackendGLWEPlaintext<B> = module.glwe_plaintext_alloc_from_infos(&layouts.glwe_pt);
                let mut coeffs = vec![0i64; n];
                // Candidates carry the `log_p`-bit message and the result is decoded at
                // `log_p`, so they must be encoded at that scale. Encoding at `base2k`
                // instead puts the message `2^(base2k - log_p)` too low, and even a
                // noiseless evaluation then decodes as 0 -- see `tests/test_vpack.rs`,
                // which is the reference for this convention.
                let log_p = params.log_p;
                let lutr = &lut;
                let glwe_layout = &layouts.glwe;
                // Trivial candidates `(0, P)`: the table is an input (§2.3.7), which is
                // also what the first tree level's credit is paid out of.
                let mut gen_ct = |c: usize, _sc: &mut poulpy_hal::layouts::ScratchArena<B>| -> BackendGLWE<B> {
                    for (j, slot) in coeffs.iter_mut().enumerate() { *slot = lutr[c * n + j] as i64; }
                    pt_c.encode_vec_i64(&coeffs, log_p.into());
                    trivial_glwe(&module, glwe_layout, &pt_c)
                };
                let mut res: BackendGLWE<B> = module.glwe_alloc_from_infos(&layouts.glwe);
                vpack_eval_streaming::<u64, B>(&module, &mut res, n_cand, &mut gen_ct,
                    FirstLevel::Streaming, &tree_sel, &rot_sel, log_n, log_table, &mut scratch.borrow());
                let mut rl: BackendLWE<B> = module.lwe_alloc_from_infos(&layouts.lwe);
                module.lwe_sample_extract(&mut rl, &res);
                let mut pt3: BackendLWEPlaintext<B> = module.lwe_plaintext_alloc(params.base2k.into(), params.k_rlwe.into());
                module.lwe_decrypt(&rl, &mut pt3, &sk.lwe, &mut scratch.borrow());
                let got3 = pt3.decode_i64(params.log_p.into()).rem_euclid(p);
                let want = (lut[k as usize] as i64).rem_euclid(p);
                if got3 == want { ok_vpack += 1; } else { println!("  vpack   k={k}: got {got3} want {want}"); }
            }
        }
        println!("N={} alpha={} d={} (Bg=2^{}, l={}, k_rlwe={}, gadget spans {} of {} bits): ohvpack {}/{}, vpack {}/{}",
            params.n_rlwe, params.alpha, params.depth, params.bg_bits, params.l, params.k_rlwe,
            params.bg_bits * params.l, params.k_rlwe,
            ok_ohvpack, idxs.len(), ok_vpack, idxs.len());
    }
}
