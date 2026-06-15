# Statistical evaluation of prompt conditions

## Descriptive accuracy

| prompt        |   accuracy |   ci_lower |   ci_upper |
|:--------------|-----------:|-----------:|-----------:|
| Zero_Shot_CoT |     0.8450 |     0.8100 |     0.8800 |
| Noise         |     0.8300 |     0.7925 |     0.8650 |
| Baseline      |     0.8200 |     0.7825 |     0.8575 |
| Structured    |     0.7925 |     0.7525 |     0.8325 |
| Yes_Man       |     0.4750 |     0.4275 |     0.5250 |

## Cochran's Q omnibus test

Cochran's Q(4) = 395.848, p = 2.195e-84.

Conclusion: reject H0. At least one prompt condition has a different probability of a correct answer.

## Pairwise exact McNemar tests with Holm-Bonferroni correction

| prompt_A   | prompt_B      |   accuracy_A |   accuracy_B |   accuracy_difference_A_minus_B |   A_correct_B_wrong |   A_wrong_B_correct |   discordant |   p_value_raw |   p_value_holm | significant_holm_0.05   |   ci_lower |   ci_upper |
|:-----------|:--------------|-------------:|-------------:|--------------------------------:|--------------------:|--------------------:|-------------:|--------------:|---------------:|:------------------------|-----------:|-----------:|
| Yes_Man    | Zero_Shot_CoT |       0.4750 |       0.8450 |                         -0.3700 |                   3 |                 151 |          154 |        0.0000 |         0.0000 | True                    |    -0.4200 |    -0.3225 |
| Noise      | Yes_Man       |       0.8300 |       0.4750 |                          0.3550 |                 145 |                   3 |          148 |        0.0000 |         0.0000 | True                    |     0.3075 |     0.4050 |
| Baseline   | Yes_Man       |       0.8200 |       0.4750 |                          0.3450 |                 144 |                   6 |          150 |        0.0000 |         0.0000 | True                    |     0.2950 |     0.3950 |
| Structured | Yes_Man       |       0.7925 |       0.4750 |                          0.3175 |                 139 |                  12 |          151 |        0.0000 |         0.0000 | True                    |     0.2675 |     0.3675 |
| Structured | Zero_Shot_CoT |       0.7925 |       0.8450 |                         -0.0525 |                   9 |                  30 |           39 |        0.0011 |         0.0064 | True                    |    -0.0825 |    -0.0225 |
| Noise      | Structured    |       0.8300 |       0.7925 |                          0.0375 |                  24 |                   9 |           33 |        0.0135 |         0.0677 | False                   |     0.0100 |     0.0675 |
| Baseline   | Structured    |       0.8200 |       0.7925 |                          0.0275 |                  22 |                  11 |           33 |        0.0801 |         0.3206 | False                   |     0.0000 |     0.0550 |
| Baseline   | Zero_Shot_CoT |       0.8200 |       0.8450 |                         -0.0250 |                   9 |                  19 |           28 |        0.0872 |         0.3206 | False                   |    -0.0500 |     0.0000 |
| Baseline   | Noise         |       0.8200 |       0.8300 |                         -0.0100 |                  11 |                  15 |           26 |        0.5572 |         0.6899 | False                   |    -0.0350 |     0.0150 |
| Noise      | Zero_Shot_CoT |       0.8300 |       0.8450 |                         -0.0150 |                  11 |                  17 |           28 |        0.3449 |         0.6899 | False                   |    -0.0425 |     0.0100 |

## Holm-significant pairwise differences

- Yes_Man vs Zero_Shot_CoT: difference A-B = -0.3700, 95% bootstrap CI [-0.4200, -0.3225], Holm p = 5.332e-40. Higher accuracy: Zero_Shot_CoT.
- Noise vs Yes_Man: difference A-B = 0.3550, 95% bootstrap CI [0.3075, 0.4050], Holm p = 2.726e-38. Higher accuracy: Noise.
- Baseline vs Yes_Man: difference A-B = 0.3450, 95% bootstrap CI [0.2950, 0.3950], Holm p = 1.671e-34. Higher accuracy: Baseline.
- Structured vs Yes_Man: difference A-B = 0.3175, 95% bootstrap CI [0.2675, 0.3675], Holm p = 1.004e-27. Higher accuracy: Structured.
- Structured vs Zero_Shot_CoT: difference A-B = -0.0525, 95% bootstrap CI [-0.0825, -0.0225], Holm p = 0.0064. Higher accuracy: Zero_Shot_CoT.