# Golden Parlay search

133 rules tested. Legs: one per game, from every spread/total/ML bet, priced at the closing line.
Each parlay risks 0.5u. Rule chosen on 2019-2021; holdout 2022-2025 judged once.

## The rule the tuning seasons picked

- 4 legs, spread, top p_win, leg EV floor 2%
- Tuning ROI +33.9% over 52 weeks
- **Holdout: 0-41, -20.5u, ROI -100.0%** (model expected +32.6%)

## How many rules were profitable

- Tuning seasons: 64 of 133
- Holdout seasons: 27 of 133
- Correlation between a rule's tuning ROI and its holdout ROI: +0.19
  (near zero means past parlay results say nothing about future ones)

## By leg count (best tuning rule for each, holdout result)

| Legs | Rule | Tuning ROI | Holdout record | Holdout ROI |
|---|---|---|---|---|
| 2 | spread, top EV, EV floor none | +21.9% | 17-55 (-5.5u) | -15.3% |
| 3 | spread, top p_win, EV floor none | +32.1% | 6-66 (-14.7u) | -40.7% |
| 4 | spread, top p_win, EV floor 2% | +33.9% | 0-41 (-20.5u) | -100.0% |
| 5 | any, top p_win, EV floor 0% | +27.3% | 4-65 (-25.4u) | -73.6% |
| dynamic | any, top p_win, EV floor 0% | +25.9% | 11-60 (-16.4u) | -46.1% |

## By market mix (best tuning rule for each, holdout result)

| Mix | Rule | Tuning ROI | Holdout ROI |
|---|---|---|---|
| any | 5 legs, top p_win, EV floor 0% | +27.3% | -73.6% |
| spread | 4 legs, top p_win, EV floor 2% | +33.9% | -100.0% |
| total | 2 legs, top p_win, EV floor 0% | +20.4% | +42.8% |
| ml | 3 legs, top p_win, EV floor 0% | +28.7% | -42.4% |
| spread+total | 5 legs, top EV, EV floor 2% | +16.0% | -100.0% |

## Reference: what a parlay costs with no edge

| Legs | ROI at -110 per leg, 50% legs |
|---|---|
| 2 | -8.9% |
| 3 | -13.0% |
| 4 | -17.0% |
| 5 | -20.8% |