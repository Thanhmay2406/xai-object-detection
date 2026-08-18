# E0-E5 pilot analysis
Source: `outputs/baseline`, `outputs/e0` ... `outputs/e5`. Metrics are from `metrics.csv`; configuration is from `experiment.json`.
Important: `MR-2_generic` is an internal diagnostic, not official CityPersons `MR^-2 Reasonable`. This is a single-seed pilot, not a paper-level statistical conclusion.
## Run validation
- baseline: method=baseline, stage=-, epochs=30, schema=nan, git=nan, odam_weight=0.2, dpga_alpha_max=nan
- E0: method=odam, stage=E0, epochs=30, schema=3.0, git=c025a840, odam_weight=0.2, dpga_alpha_max=nan
- E1: method=odam, stage=E1, epochs=30, schema=3.0, git=c025a840, odam_weight=0.2, dpga_alpha_max=nan
- E2: method=odam, stage=E2, epochs=30, schema=3.0, git=c025a840, odam_weight=0.2, dpga_alpha_max=nan
- E3: method=dpga, stage=E3, epochs=30, schema=3.0, git=c025a840, odam_weight=0.2, dpga_alpha_max=0.2
- E4: method=dpga, stage=E4, epochs=30, schema=3.0, git=c025a840, odam_weight=0.2, dpga_alpha_max=0.2
- E5: method=dpga, stage=E5, epochs=30, schema=3.0, git=c025a840, odam_weight=0.2, dpga_alpha_max=0.2

## Main metric summary
| run      |   best_AP |   best_AP_epoch |   final_AP |   best_AP50 |   best_AP50_epoch |   final_AP50 |   best_AP75 |   best_AP75_epoch |   final_AP75 |   best_MR-2_generic |   best_MR-2_generic_epoch |   final_MR-2_generic |   best_ODAM_quality |   best_ODAM_quality_epoch |   final_ODAM_quality |
|:---------|----------:|----------------:|-----------:|------------:|------------------:|-------------:|------------:|------------------:|-------------:|--------------------:|--------------------------:|---------------------:|--------------------:|--------------------------:|---------------------:|
| baseline |  0.052532 |              25 |   0.051192 |    0.154654 |                27 |     0.151409 |    0.027336 |                25 |     0.024406 |            0.957901 |                        25 |             0.959568 |          nan        |                nan        |           nan        |
| E0       |  0.030881 |              26 |   0.030608 |    0.103924 |                28 |     0.101824 |    0.013520 |                17 |     0.013109 |            0.976946 |                        27 |             0.977871 |            0.899348 |                  5.000000 |             0.873219 |
| E1       |  0.033112 |              20 |   0.032572 |    0.105642 |                28 |     0.105252 |    0.013929 |                20 |     0.013602 |            0.974737 |                        27 |             0.976019 |            0.868710 |                  5.000000 |             0.857725 |
| E2       |  0.032339 |              15 |   0.031542 |    0.105703 |                29 |     0.105703 |    0.013798 |                13 |     0.009784 |            0.977298 |                        26 |             0.977376 |            0.963196 |                  0.000000 |             0.813724 |
| E3       |  0.031363 |              23 |   0.030213 |    0.105605 |                29 |     0.105605 |    0.013802 |                15 |     0.007703 |            0.976460 |                        27 |             0.976596 |            0.870568 |                  0.000000 |             0.769537 |
| E4       |  0.033912 |              24 |   0.033334 |    0.107503 |                26 |     0.106050 |    0.013314 |                10 |     0.012775 |            0.976015 |                        27 |             0.976755 |            0.856219 |                  5.000000 |             0.788541 |
| E5       |  0.034682 |              28 |   0.034350 |    0.110107 |                28 |     0.109055 |    0.014208 |                 8 |     0.013318 |            0.974906 |                        29 |             0.974906 |            0.953563 |                  0.000000 |             0.801168 |

## Incremental deltas vs previous stage
| transition   |   delta_best_AP |   delta_final_AP |   delta_best_AP50 |   delta_final_AP50 |   delta_best_MR-2_generic |   delta_final_MR-2_generic |
|:-------------|----------------:|-----------------:|------------------:|-------------------:|--------------------------:|---------------------------:|
| E0->E1       |       +0.002231 |        +0.001964 |         +0.001718 |          +0.003428 |                 -0.002209 |                  -0.001852 |
| E1->E2       |       -0.000774 |        -0.001029 |         +0.000061 |          +0.000451 |                 +0.002561 |                  +0.001357 |
| E2->E3       |       -0.000975 |        -0.001329 |         -0.000099 |          -0.000099 |                 -0.000838 |                  -0.000780 |
| E3->E4       |       +0.002548 |        +0.003120 |         +0.001898 |          +0.000445 |                 -0.000445 |                  +0.000159 |
| E4->E5       |       +0.000770 |        +0.001017 |         +0.002604 |          +0.003005 |                 -0.001109 |                  -0.001849 |

## ODAM filtering and gradient behavior
| run      |   mean_odam_num_candidates |   mean_odam_num_kept |   mean_odam_keep_ratio |   final_odam_num_candidates |   final_odam_num_kept |   final_odam_keep_ratio |
|:---------|---------------------------:|---------------------:|-----------------------:|----------------------------:|----------------------:|------------------------:|
| baseline |                 nan        |           nan        |             nan        |                  nan        |            nan        |              nan        |
| E0       |                  21.753396 |            21.753396 |               0.937772 |                   23.256893 |             23.256893 |                0.942838 |
| E1       |                  19.600785 |            19.600785 |               0.814246 |                   23.173504 |             23.173504 |                0.943510 |
| E2       |                  19.723448 |             0.081798 |               0.002876 |                   23.307330 |              0.122394 |                0.004336 |
| E3       |                  19.657891 |             0.069514 |               0.002524 |                   23.282112 |              0.111298 |                0.003721 |
| E4       |                  19.594777 |             0.076508 |               0.002959 |                   23.166443 |              0.120040 |                0.004399 |
| E5       |                  19.779915 |             0.092636 |               0.003202 |                   23.363147 |              0.144586 |                0.004795 |

| run   |   rows |   cosine_raw_mean |   aux_to_det_raw_mean |   aux_to_det_effective_mean |   gate_mean |   effective_weight_mean |   projected_rate |   cap_active_rate |   unsafe_descent_rate |
|:------|-------:|------------------:|----------------------:|----------------------------:|------------:|------------------------:|-----------------:|------------------:|----------------------:|
| E0    |   5400 |         -0.010272 |              0.276022 |                    0.055204 |    1.000000 |                0.200000 |         0.000000 |          0.000000 |              0.319259 |
| E1    |   5400 |          0.000285 |              0.275623 |                    0.050317 |    1.000000 |                0.159805 |         0.000000 |          0.000000 |              0.273704 |
| E2    |   5400 |         -0.000631 |              0.007549 |                    0.001484 |    1.000000 |                0.159805 |         0.000000 |          0.000000 |              0.012222 |
| E3    |   5400 |          0.000278 |              0.014181 |                    0.002755 |    0.029630 |                0.005872 |         0.013333 |          0.000000 |              0.000000 |
| E4    |   5400 |          0.000758 |              0.007885 |                    0.000332 |    0.025185 |                0.004983 |         0.009259 |          0.012222 |              0.000000 |
| E5    |   5400 |          0.001442 |              0.010115 |                    0.000284 |    0.020131 |                0.003983 |         0.015185 |          0.017778 |              0.000000 |

## Reading
- Baseline Faster R-CNN has the highest detection AP in this pilot. Do not claim DPGA beats baseline from these runs.
- Within E0-E5, E5 is strongest by best/final AP and AP50; E4/E5 improve over E3, suggesting norm-cap and gate helped after projection.
- E2-E5 filtering is very strict: mean keep ratio is around 0.25%-0.32%, so most foreground candidates do not contribute to ODAM. This makes ODAM gradient much smaller and safer, but may also underuse explanation supervision.
- DPGA stages remove unsafe ODAM descent in logged diagnostics (`unsafe_descent_rate` is 0 for E3-E5), while E0/E1 have much larger unsafe rates.
- Because this is one seed and AP values are low, treat these as pilot diagnostics. For paper claims, rerun with multiple seeds and official CityPersons MR evaluation.
