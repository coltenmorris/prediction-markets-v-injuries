# Star Injury Overreaction Study

Do prediction markets overreact when a star player is ruled out? This repo backtests that idea on 13,467 settled
Kalshi game markets (NFL, NCAAF, NCAAM, NBA, NHL, MLB, 2025 season through Nov 2026) and publishes the results as an
interactive website built with Python and Plotly.

**Finding:** no. Teams whose price fell 6+ points before the game won 41.1% of the time against 41.6% implied by their
final price. Fading them (betting the opponent) does not beat Kalshi's fees in any sport over the full sample; the best
lead, NCAAF at 10+ point drops, is not statistically reliable yet.

## Run it locally
```bash
pip install -r requirements.txt
python build_site.py
# open site/index.html in a browser
```

## Publish on GitHub Pages
1. Create a new GitHub repository and push this folder to the `main` branch.
2. In the repo, open Settings > Pages and set Source to **GitHub Actions**.
3. The workflow in `.github/workflows/pages.yml` rebuilds the site on every push. The live link appears under
   Actions > the latest run, and on the Pages settings screen.

## What's in here
| Path | What it is |
|---|---|
| `build_site.py` | Reads `data/`, computes every statistic, writes `site/index.html` |
| `data/events.csv` | One row per game, from the side of the team the market marked down |
| `data/games.csv` | One row per game with Kalshi prices at 7d, 6d, 72h, 48h, 24h, 6h, 1h and the start |
| `data/nfl_star_out.csv` | NFL games with a star ruled out, matched to Kalshi prices |
| `data/miami_nd_path.csv` | Hourly Miami win price before Miami at Notre Dame, Nov 7 2026 |
| `pipeline/` | The scripts that produced the data: browser fetch engine (`eng.js`), `analyze.py`, `study.py`, `report.py`, `backtest.py`, and the NFL star table builder `build_nfl.py` |

## Definitions
- **Star player (NFL):** a QB with 60%+ of the team's pass attempts in prior games, or the WR/TE/RB leading the team in
  scrimmage yards per game at 70+, each with 3+ prior games. A star-out event is an Out or Doubtful listing with no stat line.
- **Marked-down team (all sports):** the side whose Kalshi price fell between the reference time (6 days out for NFL and
  NCAAF, 72 hours elsewhere) and the start. This captures injuries plus any other pre-game news.
- **Price:** hourly bid/ask midpoint. **ROI:** buy at the start price plus 1 cent, paying Kalshi's taker fee 0.07 x P x (1 minus P).

Data: Kalshi public API (pulled Nov 8 2026) and nflverse. Research only, not financial advice.
