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

_To be written after Part B._

## Model results (classification)

_To be written after Part C._

## Operations recommendation

_To be written after Part D._

## Production plan

_To be written after Part D._

## Limitations

_To be written after Part D._