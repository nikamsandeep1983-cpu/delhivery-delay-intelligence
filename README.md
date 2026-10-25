# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project, Artificial Intelligence and Machine Learning Certification Program (E&ICT Academy, IIT Roorkee). All data is synthetic and produced by `generate_data.py`.

## How to run

_To be written after Part D._

## Data findings

All numbers below are copied from `outputs/part_a_delay_by_group.csv`, `outputs/part_a_correlations.csv` and `outputs/part_a_checks.csv`. The modelling table has 1,200 unique shipments, of which 278 were delayed (delay rate 0.2317).

1. **Express parcels are delayed most often.** Express has a delay rate of 0.4897 (119 of 243 parcels), against 0.2056 for Standard (132 of 642) and 0.0857 for Economy (27 of 315). Express carries the tightest promise of 36 hours, so almost half of its parcels miss it.

2. **Delays climb steeply with hub tier.** The delay rate is 0.0400 for Tier 1 (16 of 400), 0.2150 for Tier 2 (86 of 400) and 0.4400 for Tier 3 (176 of 400). A Tier 3 parcel is 11 times as likely to be late as a Tier 1 parcel.

3. **Delays climb even more steeply with distance.** The delay rate is 0.0219 for the 0-200 km band (6 of 274), 0.1368 for 201-400 km (77 of 563), 0.4364 for 401-600 km (120 of 275) and 0.8523 for 601+ km (75 of 88). Beyond 600 km, more than eight in ten parcels are late.

4. **Three columns are most strongly correlated with delivery_hours.** Ranked by absolute value in `part_a_correlations.csv`, they are `promised_hours` (0.8372), `distance_km` (0.4013) and `n_stops` (0.3247). The promised time is the strongest signal because the service type sets the base delivery time, while distance and the number of stops on the rider's route add hours on top.

5. **declared_value_inr was log-transformed because it is strongly right-skewed.** Before the transform the mean is INR 1,533.05 against a median of INR 1,121.50, so a few high-value parcels pull the mean about 37% above the median; after the natural log the mean (7.0277) and median (7.0224) almost coincide, which gives the models a more symmetric input.

## Model results (regression)

All numbers below come from `outputs/part_b_metrics.csv`, `outputs/part_b_tree_depth.csv`, `outputs/part_b_gb_learning_rate.csv` and `outputs/part_b_cv.csv`.

1. **Baseline.** Predicting the training mean for every parcel gives a validation R² of −0.0212 (RMSE 18.8552 hours). It explains none of the variation in delivery time, so every real model has to beat this floor.

2. **Validation winner.** Linear regression had the lowest validation RMSE, 4.8912 hours (R² 0.9313). Gradient boosting was close behind at 4.9165, then random forest at 5.468 and the decision tree at 6.2796. On the test rows, scored once, the linear model reached an RMSE of 4.5235 hours and an R² of 0.9493.

3. **Bias and variance.** Shallow trees are high-bias: at depth 1 both errors are high (train 11.6334, val 12.1324) and almost equal, a gap of only 0.4990, so the tree is too simple to capture the pattern. Deep trees are high-variance: with no depth limit the training error falls to 0.0 but the validation error is 7.789, a gap of 7.789, and at depth 12 the gap is 6.056 (train 1.3613, val 7.4173). Depth 6 gave the lowest validation error (train 4.7114, val 6.2796), so it balances the two.

4. **Learning rate.** A learning rate of 0.1 was best (validation R² 0.9306, RMSE 4.9165). At 0.01 each tree corrects too little, so 100 trees are not enough and the model under-fits (R² 0.7127). At 1.0 each tree overshoots the correction and the fit becomes unstable (R² 0.8440).

5. **Which model to ship.** The two rankings disagree: linear regression wins on validation, but gradient boosting has the higher cross-validated R² (0.9352 against 0.9261). I would ship gradient boosting. The validation edge of linear regression is tiny, only 0.0253 hours of RMSE (4.8912 against 4.9165) on a single split of 180 rows. The cross-validated edge of gradient boosting is 0.0091 R², about two fold standard deviations (0.0044 to 0.0048), and it is averaged over five folds of the 1,020 training and validation rows, so it is the more reliable estimate. The two models have similar fold stability, so the simplicity of linear regression is its main advantage. The one weakness of this choice is that the single test score in `part_b_metrics.csv` belongs to linear regression, because the rule selects the lowest validation RMSE. Gradient boosting has no test estimate yet, so I would check its error on the first batch of new parcels before relying on it.

## Model results (classification)

All numbers below come from `outputs/part_c_models.csv`, `outputs/part_c_feature_scores.csv`, `outputs/part_c_selectk.csv`, `outputs/part_c_tuning.csv`, `outputs/part_c_thresholds.csv` and `outputs/part_c_final_test.csv`. The stratified split gives 840 training, 180 validation and 180 test parcels, with 195, 42 and 41 delayed parcels respectively.

1. **Blind baseline.** A model that predicts "not delayed" for every parcel scores an accuracy of 0.7667 on the validation rows (138 of 180 correct), which looks respectable. It is useless because it never raises an alert: its recall is 0 and all 42 delayed parcels are missed. The three real classifiers reach validation accuracies of 0.8833 to 0.9056, and their F1 scores are 0.8046 for logistic regression, 0.747 for the decision tree (depth 6) and 0.7848 for the default random forest.

2. **Noise columns.** The five bookkeeping columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) carry no signal, yet together they take 0.2245 of the forest's importance across the 17 columns, because a forest still splits on them to fit random patterns. The F-score separates them cleanly: all five rank 8th or lower (ranks 8, 12, 14, 16 and 17) with F-scores between 0.023 and 4.5847, while the seven real columns (`distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`) hold ranks 1 to 7 with F-scores from 36.7281 to 389.2131. So all five noise columns fall outside the top 7.

3. **Feature selection.** The validation F1 for the forest with k selected features is 0.6988 at k=5, 0.8333 at k=7, 0.7857 at k=10, 0.8049 at k=12 and 0.7848 at k=17. It peaks at k=7, the number of columns that actually carry signal. At k=5 the model loses `rain_flag` and `sale_event_flag` and the score drops sharply, and beyond k=7 each extra column is mostly noise, so the score falls back. I used k=7 from then on.

4. **Tuning.** Tuning did not beat the untuned forest. The grid search (27 candidates, 135 fits) reached a cross-validated accuracy of 0.8952 and a validation F1 of 0.775, and the randomized search (9 candidates, 45 fits) reached 0.8905 and 0.7848, while the untuned forest on the same seven columns scored a validation F1 of 0.8333. The cross-validated accuracies of the two searches differ by only 0.0047, so they are effectively tied. The validation set has only 42 delayed parcels, so one parcel changes recall by about 0.024 and F1 differences of 0.05 amount to a few parcels. A validation set that small cannot reliably separate configurations this close, and picking the winner from it partly rewards luck. I kept the untuned forest because it had the highest validation F1 of the three candidates, but I treat that choice as weakly supported.

5. **Alert threshold.** The operations rule is the highest threshold whose validation recall is at least 0.80, which gives **0.5**. At 0.5 the model catches 35 of the 42 delayed parcels (recall 0.8333) with a precision of 0.7955. Raising the threshold to 0.6 lifts precision to 0.8889 but recall falls to 0.7619, below the rule. Lowering it buys more recall at a cost in false alarms: at 0.4 recall is 0.881 but precision is 0.74, and at 0.1 recall is 0.9762 while precision drops to 0.4824.

6. **Test result.** At the threshold of 0.5, scored once on the test rows, the model has a precision of 0.6889, a recall of 0.7561 and an F1 of 0.7209: it caught 31 of the 41 delayed parcels and raised 14 false alarms. Test recall is lower than the 0.8333 seen on validation and below the 0.80 target. The threshold was chosen on the validation rows, so its validation recall is optimistic, and with only 41 delayed parcels a gap of about three parcels moves recall by this much. I would expect a recall of roughly 0.75 on new parcels rather than 0.83, and I would not claim the 0.80 target is met.

## Operations recommendation

_To be written after Part D._

## Production plan

_To be written after Part D._

## Limitations

_To be written after Part D._
