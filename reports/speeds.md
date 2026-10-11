# Adaptation-speed test

Tuned on 2018-2021. Holdout 2022-2025, graded at closing lines.

Per-unit parameters chosen (half-life in weeks; gap = offseason length in weeks of decay):

| Unit | Current | Per-unit |
|---|---|---|
| hl_pass_off | 16 | 16 |
| hl_pass_def | 16 | 16 |
| hl_rush_off | 16 | 16 |
| hl_rush_def | 16 | 24 |
| hl_tend | 16 | 16 |
| hl_st | 16 | 16 |
| gap_off | 8 | 12 |
| gap_def | 8 | 4 |

| Metric | current (one speed) | per-unit speeds |
|---|---|---|
| Margin MAE, tuning 2018-21 | 10.386 | 10.384 |
| Total MAE, tuning 2018-21 | 10.876 | 10.862 |
| Margin MAE, holdout | 9.872 | 9.883 |
| Total MAE, holdout | 10.329 | 10.346 |
|   closing line margin MAE | 9.536 | 9.536 |
|   closing line total MAE | 10.189 | 10.189 |
| Raw model ATS vs close | 568-542-29  (51.2%)  -19.3u  ROI -1.7%  n=1139 | 567-543-29  (51.1%)  -20.9u  ROI -1.8%  n=1139 |
| Raw model O/U vs close | 587-542-10  (52.0%)  -4.2u  ROI -0.4%  n=1139 | 581-548-10  (51.5%)  -16.0u  ROI -1.4%  n=1139 |
| Raw model O/U, weeks 1-6 | 190-179-3  (51.5%)  -4.0u  ROI -1.1%  n=372 | 187-182-3  (50.7%)  -9.8u  ROI -2.6%  n=372 |
| Blended spread bets EV>2% | 228-227-8  (50.1%)  -11.9u  ROI -2.6%  n=463 | 230-224-9  (50.7%)  -7.2u  ROI -1.5%  n=463 |
| Blended total bets EV>0 | 50-51-1  (49.5%)  -1.3u  ROI -1.3%  n=102 | 57-58-2  (49.6%)  -1.8u  ROI -1.6%  n=117 |