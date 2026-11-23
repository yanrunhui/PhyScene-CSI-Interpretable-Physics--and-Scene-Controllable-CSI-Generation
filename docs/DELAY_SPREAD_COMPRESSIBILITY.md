# Phase 1/2: paired delay-spread compressibility

The protocol uses the prepared RayVerse shards and the existing physical CSI
synthesizer. It does not load or train B3; conditional B3 generation is Phase 3
and should follow only after the paired intervention yields an interpretable result.

The system configuration is read from `artifacts/prepared_d2los/manifest.json`.
The checked-in `configs/data.yaml` currently specifies 3.5 GHz, 100 MHz,
8x8 Tx UPA, 1x1 Rx, and 64 subcarriers. The array builder enumerates antenna
positions with `meshgrid(indexing="ij")` then row-major `reshape`, so
`H[antenna, subcarrier]` maps to `[row, column, subcarrier]` by row-major
reshape. This is also the mapping in the existing `AngleDelayTransform`.

On the server, after preparing the RayVerse dataset:

```bash
python scripts/run_delay_spread_compressibility.py \
  --prepared-root artifacts/prepared_d2los \
  --output artifacts/compressibility/delay_spread \
  --phase both --phase1-count 10 --base-count 200 \
  --targets-ns 20 50 80 100 150 --seed 2026
```

Phase 1 reads 10 stored complex RayVerse CSI examples, checks the synthesizer
agrees with the stored CSI, and validates the transform, energy conservation,
best-k reconstruction, monotonicity and k99/k999 metrics. Phase 2 selects 200
LoS samples with five untruncated paths and K-factor within 10 ± 3 dB. If the
prepared training split lacks 200 eligible samples, the command reports the
rejection counts. Widen the K interval with `--k-tolerance-db`, disable that
filter with `--no-k-filter`, or set `--path-count 0`; record the change when
reporting results. `--phase 1` and `--phase 2` can be run independently.

For each base path group, Phase 2 scales delays around the earliest path while
holding powers, phases, angles, masks and LoS status fixed. It recomputes RMS
delay spread and rejects candidates whose delay span reaches 90% of the OFDM
unambiguous delay (`N/B`). With the current 100 MHz/64-subcarrier setting that
period is 640 ns. Only CSI is normalized to unit Frobenius energy before
compression. Path power and pre-normalization CSI energy are recorded.

Outputs include `phase1_per_sample_metrics.csv`,
`phase1_compression_curves.csv`, `per_sample_metrics.csv`,
`compression_curves.csv`, `paired_summary.json`, `run_metadata.json`, five
figures, and `example_angle_delay.npz`. The summary reports the arithmetic
mean of sample NMSE in dB separately from the dB of mean linear NMSE. The
paired k99 mean difference has a bootstrap 95% interval. Best-k is an oracle
analysis with sample-specific coefficient indices; no indices, quantization or
feedback bit cost is included. Modified path delays are controlled physical
interventions, not asserted to be valid rays of the original map.

The path selection uses the requested split and seeded permutation. Use the
cross-map test split only for a final validation, not for selecting target
spreads or tuning analysis decisions. In particular, inspect support and
the selection counts before interpreting the 150 ns group.
