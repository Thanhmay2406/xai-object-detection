# DPGA-ODAM Metric Report

Generated from completed run artifacts under `outputs/`.

Important notes:
- `MR-2_Reasonable`, `MR-2_Heavy`, and `MR-2_Small` remain reserved for external official CityPersons evaluator CSV input.
- `offline_best_MR-2_Reasonable`, `offline_best_MR-2_Heavy`, and `offline_best_MR-2_Small` are recomputed from saved predictions using CityPersons-style filters in the converted COCO annotations.
- `MR-2_generic` is the internal generic log-average miss-rate diagnostic from `train.py`, not the official CityPersons protocol.
- `BBox Energy Ratio` is computed from stored ODAM `dam_energy_in_gt` rows.
- `Pointing Game`, `Saliency IoU`, peak GPU memory, FPS, and parameter count are marked `NA` unless the required artifacts are supplied/stored.
- `final_gradient_norm` is marked `NA` because current gradient diagnostics store module-wise detection and ODAM norms, not the full composed gradient vector norm.

## Detection
| run | method | offline_best_MR-2_Reasonable | offline_best_MR-2_Heavy | offline_best_MR-2_Small | offline_best_AP50 | offline_best_AP | best_AP50 | best_AP | best_MR-2_generic | offline_final_MR-2_Reasonable | offline_final_AP50 | final_AP50 | final_AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | 0.896445 | 0.948183 | 0.987173 | 0.152377 | 0.0525316 | 0.154654 | 0.0525316 | 0.957901 | 0.895694 | 0.151409 | 0.151409 | 0.0511918 |
| E0 | odam | 0.937195 | 0.97049 | 0.9931 | 0.102185 | 0.0308815 | 0.103924 | 0.0308815 | 0.976946 | 0.938036 | 0.101824 | 0.101824 | 0.030608 |
| E1 | odam | 0.935443 | 0.971154 | 0.990804 | 0.102915 | 0.0331121 | 0.105642 | 0.0331121 | 0.974737 | 0.935383 | 0.105252 | 0.105252 | 0.0325716 |
| E2 | odam | 0.939502 | 0.973237 | 0.994408 | 0.103348 | 0.0323385 | 0.105703 | 0.0323385 | 0.977298 | 0.93825 | 0.105703 | 0.105703 | 0.0315423 |
| E3 | dpga | 0.939129 | 0.970973 | 0.995408 | 0.103859 | 0.0313635 | 0.105605 | 0.0313635 | 0.97646 | 0.936302 | 0.105605 | 0.105605 | 0.0302133 |
| E4 | dpga | 0.93787 | 0.971551 | 0.994742 | 0.107365 | 0.0339116 | 0.107503 | 0.0339116 | 0.976015 | 0.939379 | 0.10605 | 0.10605 | 0.0333337 |
| E5 | dpga | 0.936511 | 0.971077 | 0.996721 | 0.110107 | 0.034682 | 0.110107 | 0.034682 | 0.974906 | 0.935592 | 0.109055 | 0.109055 | 0.0343505 |

## Gradient
| run | cosine_similarity_mean | cosine_similarity_median | gradient_conflict_rate | gradient_norm_ratio_mean | projection_rate | norm_cap_rate | final_aux_gradient_norm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | NA | NA | NA | NA | NA | NA | NA |
| E0 | -0.0118269 | 0 | 0.881356 | 0.317808 | 0 | 0 | 0.240162 |
| E1 | 0.00037716 | 0 | 0.859917 | 0.364974 | 0 | 0 | 0.249364 |
| E2 | -0.0141966 | 0 | 0.73913 | 0.169849 | 0 | 0 | 0.192072 |
| E3 | 0.00631354 | 0 | 0.608696 | 0.321751 | 0.608696 | 0 | 0.29507 |
| E4 | 0.0161113 | 0 | 0.5 | 0.167629 | 0.5 | 0.5 | 0.0458144 |
| E5 | 0.0246357 | 0 | 0.590164 | 0.172849 | 0.590164 | 0.590164 | 0.0375507 |

## XAI / ODAM Quality
| run | bbox_energy_ratio_mean | bbox_energy_ratio_final_epoch | pointing_game | saliency_iou | detection_match_iou_mean |
| --- | --- | --- | --- | --- | --- |
| baseline | NA | NA | NA | NA | NA |
| E0 | 0.870725 | 0.873219 | NA | NA | 0.61579 |
| E1 | 0.853805 | 0.857725 | NA | NA | 0.615978 |
| E2 | 0.81773 | 0.813724 | NA | NA | 0.61874 |
| E3 | 0.773562 | 0.769537 | NA | NA | 0.61818 |
| E4 | 0.789473 | 0.788541 | NA | NA | 0.616554 |
| E5 | 0.804885 | 0.801168 | NA | NA | 0.616902 |

## Computational Cost
| run | time_per_epoch_mean_s | time_total_s | peak_gpu_memory_gb | fps | num_parameters |
| --- | --- | --- | --- | --- | --- |
| baseline | 746.284 | 22388.5 | NA | NA | NA |
| E0 | 693.245 | 20797.4 | NA | NA | NA |
| E1 | 661.595 | 19847.8 | NA | NA | NA |
| E2 | 638.619 | 19158.6 | NA | NA | NA |
| E3 | 956.15 | 28684.5 | NA | NA | NA |
| E4 | 993.81 | 29814.3 | NA | NA | NA |
| E5 | 1034.38 | 31031.5 | NA | NA | NA |
