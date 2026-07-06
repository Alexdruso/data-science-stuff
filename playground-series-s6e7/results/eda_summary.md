# S6E7 EDA summary

train: (690088, 15)   test: (295753, 14)

## Class balance
- at-risk: 592561 (85.9%)
- unhealthy: 57724 (8.4%)
- fit: 39803 (5.8%)

## Null rate (train / test)
- sleep_duration: 11.0% / 11.0%
- heart_rate: 1.1% / 1.1%
- bmi: 2.0% / 2.0%
- calorie_expenditure: 7.7% / 7.7%
- step_count: 2.0% / 2.0%
- exercise_duration: 1.0% / 1.0%
- water_intake: 6.3% / 6.3%
- diet_type: 1.0% / 1.0%
- stress_level: 12.0% / 12.0%
- sleep_quality: 8.5% / 8.5%
- physical_activity_level: 5.3% / 5.3%
- smoking_alcohol: 4.1% / 4.1%
- gender: 3.1% / 3.1%

## Numeric features — mean by class (train) | train mean / test mean
- sleep_duration: at-risk=7.09 fit=7.95 unhealthy=5.37 | 6.99 / 6.99
- heart_rate: at-risk=75.10 fit=74.80 unhealthy=75.26 | 75.10 / 75.08
- bmi: at-risk=22.95 fit=21.83 unhealthy=24.12 | 22.98 / 22.98
- calorie_expenditure: at-risk=2214.94 fit=2363.99 unhealthy=2245.42 | 2226.08 / 2225.51
- step_count: at-risk=8406.71 fit=11651.31 unhealthy=8670.23 | 8615.95 / 8626.63
- exercise_duration: at-risk=37.96 fit=50.04 unhealthy=39.04 | 38.75 / 38.80
- water_intake: at-risk=2.19 fit=2.18 unhealthy=2.19 | 2.19 / 2.19

## Categorical — value share within each class (train)
### diet_type  (train share / test share)
- balanced: 32.9% / 33.0% | within-class: at-risk=32.6% fit=34.9% unhealthy=34.4%
- non-veg: 32.6% / 32.5% | within-class: at-risk=32.9% fit=29.1% unhealthy=31.4%
- veg: 33.5% / 33.5% | within-class: at-risk=33.5% fit=35.1% unhealthy=33.3%
### stress_level  (train share / test share)
- high: 25.8% / 25.6% | within-class: at-risk=21.5% fit=1.6% unhealthy=85.8%
- low: 24.3% / 24.3% | within-class: at-risk=22.5% fit=84.5% unhealthy=0.8%
- medium: 37.9% / 38.1% | within-class: at-risk=43.9% fit=2.0% unhealthy=1.4%
### sleep_quality  (train share / test share)
- average: 31.0% / 30.9% | within-class: at-risk=31.0% fit=30.8% unhealthy=30.9%
- good: 29.8% / 29.8% | within-class: at-risk=30.8% fit=42.2% unhealthy=10.9%
- poor: 30.7% / 30.9% | within-class: at-risk=29.7% fit=18.5% unhealthy=49.9%
### physical_activity_level  (train share / test share)
- active: 30.8% / 30.0% | within-class: at-risk=26.7% fit=91.7% unhealthy=31.4%
- moderate: 32.0% / 32.3% | within-class: at-risk=34.0% fit=1.7% unhealthy=32.7%
- sedentary: 31.8% / 32.4% | within-class: at-risk=34.0% fit=1.3% unhealthy=30.6%
### smoking_alcohol  (train share / test share)
- no: 31.8% / 32.0% | within-class: at-risk=32.2% fit=43.4% unhealthy=20.8%
- occasional: 31.6% / 31.7% | within-class: at-risk=31.6% fit=31.7% unhealthy=31.6%
- yes: 32.4% / 32.2% | within-class: at-risk=32.1% fit=20.8% unhealthy=43.4%
### gender  (train share / test share)
- female: 32.5% / 29.2% | within-class: at-risk=32.4% fit=32.1% unhealthy=33.2%
- male: 34.5% / 34.6% | within-class: at-risk=34.4% fit=37.4% unhealthy=33.1%
- other: 30.0% / 33.1% | within-class: at-risk=30.1% fit=27.5% unhealthy=30.5%

## Missingness vs target — null rate of each feature within each class
- sleep_duration: at-risk=11.0% fit=10.9% unhealthy=11.0%
- heart_rate: at-risk=1.2% fit=1.1% unhealthy=1.0%
- bmi: at-risk=2.1% fit=2.9% unhealthy=0.7%
- calorie_expenditure: at-risk=7.7% fit=7.7% unhealthy=7.7%
- step_count: at-risk=2.0% fit=2.0% unhealthy=2.0%
- exercise_duration: at-risk=1.0% fit=1.0% unhealthy=1.1%
- water_intake: at-risk=6.3% fit=6.3% unhealthy=6.5%
- diet_type: at-risk=1.0% fit=0.9% unhealthy=1.0%
- stress_level: at-risk=12.0% fit=12.0% unhealthy=12.0%
- sleep_quality: at-risk=8.5% fit=8.5% unhealthy=8.3%
- physical_activity_level: at-risk=5.3% fit=5.3% unhealthy=5.3%
- smoking_alcohol: at-risk=4.1% fit=4.1% unhealthy=4.2%
- gender: at-risk=3.1% fit=3.0% unhealthy=3.2%

## Adversarial validation (train=0 vs test=1)
- adversarial AUC: 0.6521  (SHIFT — inspect top features)
- top discriminating features: bmi(2622), step_count(2313), water_intake(2210), exercise_duration(2096), sleep_duration(2056), calorie_expenditure(1927), heart_rate(1718), physical_activity_level(953)
