# Key Number board: design

The board's visual system (user, 2026-10-20: "more motion, color, logos, bubbly, use the space, choose the best
pick fast"), built to Emil Kowalski's design-engineering rules, Impeccable's design laws and Taste Skill's
app-screen rules. Where those conflict with the user's brief, the brief wins (Impeccable: "the brief wins").

## What the screen is for

- **Mode:** Operate (Impeccable) / app screen (Taste). People work in it daily; scanability first.
- **Scene:** a bettor at the kitchen table at 10:30am when picks lock, and on the couch at night with the
  games on, choosing two or three bets in under a minute on a laptop or a phone. Day and night: the board
  follows the system theme (no toggle; team colors are re-picked when the theme changes).
- **Question of the landing view (Best bets):** which picks are worth betting right now? The biggest block
  answers it: one ranked list of every pick, game lines and props together, with filters, and the main job
  (add to the slip) one click away.
- **The finding** sits under the title: how many picks clear the bar today and the best of them.
- **Face:** the teams' own logos and colors as identity, the key numbers 3 and 7 as the mark, market numbers
  in a mono like a ticket.
- **Signature element:** the line-shop strip: where Pinnacle's fair number and each book's line sit on one
  axis, so the value of a price is seen, not read.

## The pieces

- **Best bets** (`#best`, `#nba-best`, the landing view; no address opens the sport looked at last): every pick
  of the week or day in one list, game lines and props, filtered (type, value only, not started, search) and
  sorted (best value = EV relative to the payout, as Picks of the Week/Day rank; most likely; soonest). A row:
  the logo or headshot, the bet, its flags, the price and book, the first context note, EV, the win-chance ring
  (its tick is the break-even rate) and the add button. A row opens in place for every note, the line-shop
  strip, the last games against the line and the checks. Parlays sit above the list, addable whole.
- **The slip**: the right rail from 1240px (with the season at a glance under it), a button and a bottom sheet
  (native `<dialog>`) below that. Prices the picks as a parlay when they are from different games (the hit
  chance multiplies, so it says so), the straights' expected units, copy as text. localStorage only; a pick
  drops off 12 hours after its start.
- **Team marks**: logos from ESPN's image service (`logo`/`color2` in both payloads' `teams`, `img` headshots on
  props players). Any image that fails falls back to the colored abbreviation or initials, so a host that
  blocks outside images still reads.
- **Public betting** (game cards, both sports): per market a split bar for the share of bets and a thinner one for
  the share of money, away (or over) on the left and home (or under) on the right, in the teams' colors; the side
  with 70%+ of bets in bold. Context only, so it sits inside the lines panel, never beside the picks.
- **Team colors on data** (win-chance bars, chart bars, the line chart): a color that would not read on the
  theme's surface is lightened or swapped for the team's second color when that has real color; two teams that
  would read as one fall back to the board's own pair.

## Tokens

- **Color strategy:** full palette for data roles only, neutral chrome. Value green (+EV, wins), loss red,
  featured gold (Picks of the Day/Week, Golden Parlay), team colors on matchup data. The primary action is
  near-black (white on dark). OKLCH, neutrals tinted 0.003-0.012 toward 250-260. Dark is a true neutral
  (no navy, no slate).
- **Type:** Figtree (UI and text, tabular figures), JetBrains Mono for market numbers only (lines, prices).
  Sizes 12 / 13 / 14 / 16 / 20 / 28, 40 for one hero figure; sentence case everywhere; no tracked uppercase.
- **Shape:** soft, chosen once: cards 20px, inner panels 14px, chips and controls fully round.
- **Surfaces:** a border or a tone step, never both with a shadow; shadows only on what floats (tooltips,
  the slip sheet, toasts, a hovered card).

## Motion

| Token | Value | Used for |
| --- | --- | --- |
| `--ease-out` | cubic-bezier(0.23, 1, 0.32, 1) | entering, presses, reveals |
| `--ease-in-out` | cubic-bezier(0.77, 0, 0.175, 1) | indicators moving on screen |
| `--ease-drawer` | cubic-bezier(0.32, 0.72, 0, 1) | the slip sheet on phones |

Press 140ms (scale 0.97), hover 160ms (pointer devices only), tooltip 125ms, accordion 240ms, view change
220ms, sheet 400ms, toast 400ms in / 200ms out (ease, Sonner). Charts draw once when first seen
(explanation, 600ms). Nothing animates from scale(0); transforms and opacity only (height only for
accordions). `prefers-reduced-motion`: fades stay, travel goes.

## Bans (checked by eye on every change)

No glass or backdrop blur, no gradient text or gradient fills (one area fade under a line chart is the
exception), no icon in a tinted circle, no sparkline on a figure, no side-stripe borders, no uppercase tracked
labels, no em dashes in copy, no emoji.
