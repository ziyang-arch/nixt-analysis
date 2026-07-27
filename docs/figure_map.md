# Paper figure ↔ script mapping

All scripts live in `analysis/` and read parquet from `$NIXT_ROOT/data/*-analysis/parquet_files/`
(produced by `exporter/perf_summary_exporter.py`). Reference copies of every generated figure,
exactly as they appear in the camera-ready paper, are in `expected_output/`.

## Workload summary & scaling (Case Study)

`summary_all_paperplot.py`, `summary_scaling_paperplot.py`
- `340b_2048_busbw_percentiles.pdf`
- `340b_2048_bytes_by_coll_commtype.pdf`
- `340b_2048_top_msg_sizes_by_bytes.pdf`
- `compare_15b_340b_2048_topmsg_and_busbw.pdf`
- `compare_modelsize_gpus2048_busbw_by_coll.pdf`
- `compare_modelsize_gpus2048_bytes_by_coll.pdf`
- `scaling_15b_busbw_by_coll_vs_gpus.pdf`
- `scaling_15b_bytes_by_coll_vs_gpus.pdf`
- `scaling_15b_total_bytes_vs_gpus.pdf`

## Measurement vs. Configuration / Counter (taxonomy case studies)

`Measurement_Config_paperplot.py`, `Measurement_Counter_paperplot.py`,
`Measurement_Identifier_paperplot.py`
- `exp_cdf_busbw_large_by_config_{15b,340b}_gpus2048_healthy_*.pdf`
- `exp_cdf_exec_small_by_config_{15b,340b}_gpus2048_healthy_*.pdf`
- `exp_situation_msgsize_count_heatmap_{15b,340b}_gpus2048_healthy_*.pdf`

## Spatial / temporal variability (range-bar analyses)

`Spatial_RangeBar_paperplot.py`
- `spatial_percommrank_rangebar_340b_gpus2048_*.pdf`
- `spatial_perhost_rangebar_340b_gpus2048_*.pdf`
- `spatial_perid_rangebar_340b_gpus2048_*.pdf`

`Temporal_RangeBar_paperplot.py`
- `temporal_percollsnbin_rangebar_340b_gpus2048_*.pdf`
- `temporal_pertimebin_rangebar_340b_gpus2048_*.pdf`

## Straggler case study

`Straggler_RangeBar_paperplot.py`
- `straggler_perid_rangebar_15b_mixedAG24MB.pdf`
- `straggler_perhost_rangebar_15b_mixedAG24MB.pdf`
- `straggler_percollsnbin_rangebar_15b_mixedAG24MB.pdf`
- `straggler_pertimebin_rangebar_15b_mixedAG24MB.pdf`

## Training vs. microbenchmark comparison (nccl-tests)

`nccl_tests_comparison_paperplot.py`
- `exp_ecdf_busbw_training_vs_ncclperf_340b_2048.pdf`

Note: this script additionally expects nccl-tests sweep outputs under
`$NIXT_ROOT/sup_data/NCCL2.30.4/` (generate with nccl-tests on your own cluster).

## Path conventions

Every script resolves its inputs/outputs relative to `NIXT_ROOT`
(environment variable; defaults to the repository root):
- input parquet: `$NIXT_ROOT/data/<experiment>-analysis/parquet_files/*.parquet`
- output figures: `$NIXT_ROOT/figures/`
