use crate::arch::*;
use crate::trainer_common::{BatchData, PrecisionFlags};
use gpu_runtime::CudaContext;
use nnue_train::optimizer::OptimizerKind;

const TOL: f32 = 1e-5;

fn deterministic_floats(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 17) as f32 - 8.0) / (100.0 * scale))
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tolerance, "{label}: {a} != {e}");
    }
}

#[test]
fn layerstack_factorized_warm_start_preserves_weights_and_trains()
-> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_layerstack::{GpuTrainer, OptimGroupConfig};
    use nnue_train::init::LayerStackInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let base_spec = FeatureSet::HalfKaHmMerged.spec();
    let factorized = base_spec.with_ft_factorize();
    let ft_out = 128;
    let buckets = 8;
    for with_psqt in [false, true] {
        for mixed_precision in [false, true] {
            let precision = PrecisionFlags {
                ft_fp16: mixed_precision,
                ft_fp16_out: mixed_precision,
                fp16_opt_state: mixed_precision,
                tf32: mixed_precision,
            };
            let psqt = with_psqt.then(|| deterministic_floats(base_spec.ft_in() * buckets, 7.0));
            let new_trainer = |spec| {
                GpuTrainer::new(
                    &ctx,
                    SMOKE_BATCH,
                    ft_out,
                    DEFAULT_L1_OUT,
                    DEFAULT_L2_OUT,
                    buckets,
                    nnue_train::dataloader::BucketMode::Progress8KpAbs,
                    precision,
                    spec,
                    OptimizerKind::Ranger,
                    OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
                    None,
                    psqt.as_deref(),
                    &LayerStackInit::default_uniform(),
                )
            };
            let mut base = new_trainer(base_spec)?;
            let original = base.to_layerstack_weights()?;
            let mut candidate = new_trainer(factorized)?;
            candidate.load_layerstack_weights(&original)?;
            let loaded = candidate.to_layerstack_weights()?;
            assert_eq!(loaded.feature_set, base_spec);
            let mut expected_bytes = Vec::new();
            let mut actual_bytes = Vec::new();
            if with_psqt {
                original.save_quantised(&mut expected_bytes, None)?;
                loaded.save_quantised(&mut actual_bytes, None)?;
            } else {
                nnue_format::save_tanuki_sfnnwop1536(&mut expected_bytes, &original, 300.0)?;
                nnue_format::save_tanuki_sfnnwop1536(&mut actual_bytes, &loaded, 300.0)?;
            }
            assert_eq!(
                actual_bytes, expected_bytes,
                "warm-start export must be identical"
            );
            let (step, groups) = candidate.raw_checkpoint_state_to_host()?;
            assert_eq!(step, 0);
            for (name, (weights, m, v, slow)) in &groups {
                assert_eq!(weights.len(), m.len(), "{name}");
                assert_eq!(weights.len(), v.len(), "{name}");
                assert_eq!(
                    weights, slow,
                    "{name}: lookahead must anchor loaded weights"
                );
                assert!(
                    m.iter().chain(v).all(|&x| x == 0.0),
                    "{name}: moments reset"
                );
                let real_len = match *name {
                    "ft_w" => Some(base_spec.ft_in() * ft_out),
                    "psqt_w" => Some(base_spec.ft_in() * buckets),
                    _ => None,
                };
                if let Some(real_len) = real_len {
                    assert!(weights.len() > real_len);
                    assert!(weights[real_len..].iter().all(|&x| x == 0.0));
                }
            }
            drop(groups);
            let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, base_spec);
            batch.score.fill(200.0);
            batch.wdl.fill(0.8);
            let expected = base.validate(&batch.as_ref(), WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
            let observed = candidate.validate(&batch.as_ref(), WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
            assert_close(
                "warm-start predictions",
                &observed.net_output,
                &expected.net_output,
                TOL,
            );
            for _ in 0..=RANGER_K {
                let loss = candidate.step(&batch.as_ref(), 1e-3, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
                assert!(loss.is_finite());
            }
            candidate.assert_all_weights_finite()?;
            let (step, groups) = candidate.raw_checkpoint_state_to_host()?;
            assert_eq!(step, RANGER_K + 1);
            let (_, (weights, _, _, _)) = groups.iter().find(|(name, _)| *name == "ft_w").unwrap();
            assert!(
                weights[base_spec.ft_in() * ft_out..]
                    .iter()
                    .any(|&x| x != 0.0)
            );
            let mut malformed = original.clone();
            malformed.ft_w.pop();
            assert!(candidate.load_layerstack_weights(&malformed).is_err());
            malformed = original.clone();
            malformed.feature_set = factorized;
            assert!(candidate.load_layerstack_weights(&malformed).is_err());
            malformed = original.clone();
            malformed.num_buckets += 1;
            assert!(candidate.load_layerstack_weights(&malformed).is_err());
            let (after_step, after_groups) = candidate.raw_checkpoint_state_to_host()?;
            assert_eq!(step, after_step);
            assert_eq!(
                groups, after_groups,
                "invalid import must not alter trainer state"
            );
        }
    }
    Ok(())
}
