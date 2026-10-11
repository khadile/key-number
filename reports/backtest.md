# Backtest report

Parameters tuned on 2018-2021; 2022+ is holdout.
Graded at nflverse closing lines and prices. Ratings params: `{'half_life': 16.0, 'gap': 8.0, 'alpha_play': 200.0, 'alpha_game': 16.0, 'qb_half_life': 250.0, 'qb_gap': 0.0, 'alpha_qb': 250.0, 'hl_pass_off': None, 'hl_pass_def': None, 'hl_rush_off': None, 'hl_rush_def': None, 'hl_tend': None, 'hl_st': None, 'gap_off': None, 'gap_def': None}`, points ridge 500.0.

## 1. Accuracy vs the closing line

| Seasons | Games | Margin MAE model | Margin MAE close | Total MAE model | Total MAE close |
|---|---|---|---|---|---|
| Tuning 2018-2021 | 1088 | 10.386 | 10.135 | 10.876 | 10.655 |
| Holdout 2022-2026 | 1203 | 9.856 | 9.541 | 10.383 | 10.218 |

Per season:

| Season | Margin model | Margin close | Total model | Total close |
|---|---|---|---|---|
| 2018 | 10.21 | 9.87 | 11.28 | 10.72 |
| 2019 | 10.46 | 10.18 | 10.77 | 10.78 |
| 2020 | 10.02 | 9.79 | 10.73 | 10.30 |
| 2021 | 10.83 | 10.67 | 10.73 | 10.81 |
| 2022 | 9.06 | 8.78 | 10.71 | 10.40 |
| 2023 | 10.31 | 9.98 | 10.51 | 10.17 |
| 2024 | 10.19 | 9.70 | 9.81 | 9.77 |
| 2025 | 9.93 | 9.67 | 10.28 | 10.42 |
| 2026 | 9.59 | 9.63 | 11.33 | 10.73 |

## 2. Raw model vs closing spread, by disagreement (Sam-style)

Side = whichever side the raw model margin favors vs the closing spread. Holdout only.

| Gap (pts) | Record |
|---|---|
| 0-1 | 172-165-13  (51.0%)  -7.4u  ROI -2.1%  n=350 |
| 1-2 | 147-136-10  (51.9%)  -0.6u  ROI -0.2%  n=293 |
| 2-3 | 121-97-6  (55.5%)  +14.6u  ROI +6.5%  n=224 |
| 3-5 | 115-122-5  (48.5%)  -15.9u  ROI -6.6%  n=242 |
| 5-+ | 46-48-0  (48.9%)  -6.0u  ROI -6.4%  n=94 |
| all | 601-568-34  (51.4%)  -15.4u  ROI -1.3%  n=1203 |

## 3. How much the model is trusted vs the market (blend weights)

Fit on earlier seasons' out-of-sample predictions: `result = a + b*model + c*line`.
b near 0 and c near 1 means the model adds nothing the line doesn't already have.

| Season | margin b (model) | margin c (line) | total b (model) | total c (line) | sigma margin | sigma total |
|---|---|---|---|---|---|---|
| 2019 | 0.66 | 0.72 | -0.59 | 1.38 | 13.0 | 13.6 |
| 2020 | 0.47 | 0.81 | -0.19 | 1.14 | 13.0 | 13.5 |
| 2021 | 0.47 | 0.77 | 0.01 | 1.07 | 12.9 | 13.4 |
| 2022 | 0.50 | 0.74 | 0.04 | 1.00 | 13.0 | 13.4 |
| 2023 | 0.25 | 0.83 | 0.06 | 0.98 | 12.8 | 13.3 |
| 2024 | 0.22 | 0.85 | 0.04 | 0.97 | 12.8 | 13.3 |
| 2025 | 0.26 | 0.86 | 0.06 | 0.95 | 12.8 | 13.2 |
| 2026 | 0.22 | 0.89 | 0.15 | 0.88 | 12.7 | 13.2 |

## 4. Blended bets by EV threshold (holdout, closing prices)

| Market | EV > 0% | EV > 2% | EV > 4% | EV > 6% |
|---|---|---|---|---|
| spread | 269-290-25  (48.1%)  -34.5u  ROI -5.9%  n=584 | 171-201-19  (46.0%)  -37.2u  ROI -9.5%  n=391 | 115-137-13  (45.6%)  -26.5u  ROI -10.0%  n=265 | 73-110-8  (39.9%)  -39.4u  ROI -20.6%  n=191 |
| total | 96-107-1  (47.3%)  -14.1u  ROI -6.9%  n=204 | 33-31-1  (51.6%)  +1.6u  ROI +2.4%  n=65 | 9-6-1  (60.0%)  +2.9u  ROI +18.4%  n=16 | 1-1-0  (50.0%)  -0.0u  ROI -1.0%  n=2 |
| ml | 320-319-3  (50.1%)  +10.4u  ROI +1.6%  n=642 | 232-237-3  (49.5%)  +10.2u  ROI +2.2%  n=472 | 140-172-2  (44.9%)  -13.0u  ROI -4.1%  n=314 | 106-128-2  (45.3%)  -4.4u  ROI -1.9%  n=236 |

## 5. Calibration of bets taken (EV > 0, holdout)

| Model said | Bets | Actually won (excl. pushes) |
|---|---|---|
| 50%-52% | 308 | 49.4% |
| 52%-54% | 184 | 52.2% |
| 54%-56% | 81 | 38.3% |
| 56%-58% | 49 | 32.7% |
| 58%-60% | 22 | 40.9% |
| 60%-62% | 7 | 85.7% |

## 6. Picks of the Week and Golden Parlay (holdout)

| EV floor | Picks of the Week | Golden Parlay (0.5u) |
|---|---|---|
| 0% | 167-247-9  (40.3%)  -46.6u  ROI -11.0%  n=423 | 6-79  -16.7u  n=85 |
| 2% | 156-228-9  (40.6%)  -41.1u  ROI -10.5%  n=393 | 7-75  -13.7u  n=82 |
| 4% | 123-188-7  (39.5%)  -33.8u  ROI -10.6%  n=318 | 6-68  -14.8u  n=74 |

## 7. Pre-declared segments (holdout spread bets, EV > 2%)

- Weeks 1-4: 42-40-7  (51.2%)  +0.0u  ROI +0.0%  n=89
- Weeks 5-18: 125-154-12  (44.8%)  -34.1u  ROI -11.7%  n=291
- Playoffs: 4-7-0  (36.4%)  -3.2u  ROI -28.9%  n=11

Breakeven at -110 is 52.4%. Results within a few points of that over a few hundred bets are noise.