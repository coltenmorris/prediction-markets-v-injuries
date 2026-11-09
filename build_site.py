"""Build the static website for the Star Injury Overreaction Study.

Reads the CSVs in data/ and writes site/index.html with interactive Plotly charts.
Run:  python build_site.py   then open site/index.html
"""
from pathlib import Path
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

ROOT = Path(__file__).parent
# Find each CSV in data/ first, then the repo root (web uploads sometimes flatten folders or mix both).
def find(name):
    for d in (ROOT / "data", ROOT):
        if (d / name).exists():
            return d / name
    raise SystemExit(f"Missing {name}: upload it into data/ (or the repo root).")

OUT = ROOT / "site"
ORDER = ["NCAAF", "NCAAM", "NFL", "NBA", "NHL", "MLB"]
INK, MUTED, GRID = "#1d2a3a", "#8a94a3", "#e6e9ee"
ACCENT, ACCENT2, BAD, GOOD = "#c2410c", "#0f766e", "#b42318", "#067647"

ev = pd.read_csv(find("events.csv"), parse_dates=["game_date_et"])
games = pd.read_csv(find("games.csv"))
nfl = pd.read_csv(find("nfl_star_out.csv"))
col = pd.read_csv(find("college_star_out.csv"))
mia = pd.read_csv(find("miami_nd_path.csv"))


# ---------------------------------------------------------------- statistics
def stats(df: pd.DataFrame) -> dict:
    n = len(df)
    w, pc, pr = df.won.mean(), df.price_start.mean(), df.price_ref.mean()
    se = np.sqrt((df.price_start * (1 - df.price_start)).sum()) / n
    z = (w - pc) / se
    return dict(n=n, drop=df.drop_pts.mean(), p_ref=pr, p_start=pc, win=w,
                edge=(w - pc) * 100, lo=(w - pc - 1.96 * se) * 100, hi=(w - pc + 1.96 * se) * 100,
                p=2 * (1 - norm.cdf(abs(z))), kept=(w - pr) / (pc - pr) if pc != pr else np.nan,
                back_roi=df.roi_back_marked_down.mean(), fade_roi=fade_roi(df))


def fade_roi(df: pd.DataFrame, fee=0.07, slip=0.01) -> float:
    """Buy the opponent of the marked-down team at the start: +1c over mid, Kalshi taker fee."""
    a = ((1 - df.price_start) + slip).clip(upper=0.99)
    f = fee * a * (1 - a)
    pnl = (1 - df.won) - a - f
    return pnl.sum() / (a + f).sum()


def fade_pnl(df, fee=0.07, slip=0.01):
    a = ((1 - df.price_start) + slip).clip(upper=0.99)
    f = fee * a * (1 - a)
    return (1 - df.won) - a - f, a + f


big = ev[ev.drop_pts >= 6]
by_sport = pd.DataFrame([dict(sport=s, **stats(big[big.sport == s])) for s in ORDER])
head = stats(big)

# backtest grid: sport x threshold, with chronological holdout
bt = []
for s in ORDER:
    for thr in (6, 10, 15):
        d = ev[(ev.sport == s) & (ev.drop_pts >= thr)].sort_values("game_date_et")
        if len(d) < 15:
            continue
        cut = d.game_date_et.quantile(0.6)
        bt.append(dict(sport=s, thr=thr, n=len(d), roi=fade_roi(d),
                       first=fade_roi(d[d.game_date_et <= cut]), last=fade_roi(d[d.game_date_et > cut])))
bt = pd.DataFrame(bt)

# late moves (final 24h) need price_24h
g24 = ev[ev.price_24h.between(0.02, 0.98)].copy()
# re-orient on the 24h -> start move
down = g24.price_start < g24.price_24h
late = pd.DataFrame({"sport": g24.sport,
                     "price_ref": np.where(down, g24.price_24h, 1 - g24.price_24h),
                     "price_start": np.where(down, g24.price_start, 1 - g24.price_start),
                     "won": np.where(down, g24.won, 1 - g24.won)})
late["drop_pts"] = ((late.price_ref - late.price_start) * 100).round(6)
late["roi_back_marked_down"] = 0.0
late = late[late.drop_pts >= 5]
late_sport = pd.DataFrame([dict(sport=s, **stats(late[late.sport == s])) for s in ORDER if (late.sport == s).sum() >= 10])

# calibration (all games)
g = games[games.price_start.between(0.02, 0.98)].copy()
g["bin"] = (g.price_start * 10).astype(int)
calib = g.groupby(["sport", "bin"]).agg(price=("price_start", "mean"), win=("team_won", "mean"), n=("ticker", "size")).reset_index()


# ---------------------------------------------------------------- charts
def base_layout(fig, height=360, **kw):
    kw.setdefault("showlegend", False)
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=INK),
                      hoverlabel=dict(bgcolor="white", font_size=12), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID)
    return fig


def dot_ci(df, value, lo, hi, ref, fmt, xtitle, label=lambda r: r.sport):
    df = df.iloc[::-1]
    colors = [ACCENT if str(s).startswith("NCAA") else INK for s in df.sport]
    fig = go.Figure()
    for (_, r), c in zip(df.iterrows(), colors):
        fig.add_trace(go.Scatter(x=[r[lo], r[hi]], y=[label(r)] * 2, mode="lines",
                                 line=dict(color=c, width=3), opacity=0.35, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df[value], y=[label(r) for _, r in df.iterrows()], mode="markers+text",
        marker=dict(size=12, color=colors), text=[fmt(v) for v in df[value]], textposition="middle right",
        textfont=dict(size=12), cliponaxis=False,
        customdata=np.stack([df.n, df.win * 100, df.p_start * 100, df.p], axis=1),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} teams<br>Won %{customdata[1]:.1f}% vs priced %{customdata[2]:.1f}%"
                      "<br>p = %{customdata[3]:.3f}<extra></extra>"))
    fig.add_vline(x=ref, line_dash="dash", line_color=MUTED)
    base_layout(fig, height=60 + 44 * len(df))
    fig.update_xaxes(title=xtitle)
    return fig


def html_fig(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, default_width="100%",
                       config={"displayModeBar": False, "responsive": True})


f_edge = dot_ci(by_sport, "edge", "lo", "hi", 0, lambda v: f"{v:+.1f}", "Win rate minus price at start (points)",
                label=lambda r: f"{r.sport} ({r.n:,})")
f_late = dot_ci(late_sport, "edge", "lo", "hi", 0, lambda v: f"{v:+.1f}", "Win rate minus price at start (points)",
                label=lambda r: f"{r.sport} ({r.n:,})")

# backtest bars with threshold buttons
f_bt = go.Figure()
thr_list = [6, 10, 15]
for i, thr in enumerate(thr_list):
    d = bt[bt.thr == thr].set_index("sport").reindex(ORDER).dropna(subset=["n"])
    f_bt.add_trace(go.Bar(x=d.index, y=d.roi * 100, visible=(i == 0),
                          marker_color=[GOOD if v > 0 else BAD for v in d.roi], name=f"{thr}+",
                          customdata=np.stack([d.n, d["first"] * 100, d["last"] * 100], axis=1),
                          text=[f"{v:+.1f}%" for v in d.roi * 100], textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{x}</b><br>%{customdata[0]:,} bets<br>ROI %{y:+.1f}%"
                                        "<br>First 60%: %{customdata[1]:+.1f}%<br>Last 40%: %{customdata[2]:+.1f}%<extra></extra>"))
base_layout(f_bt, height=380)
f_bt.update_yaxes(title="ROI per dollar staked (%)", zeroline=True)
f_bt.update_layout(margin=dict(t=50), updatemenus=[dict(
    type="buttons", direction="right", x=0, y=1.16, xanchor="left", showactive=True,
    bgcolor="white", bordercolor=GRID, font=dict(size=12),
    buttons=[dict(label=f"Drop {t}+ pts", method="update",
                  args=[{"visible": [j == i for j in range(len(thr_list))]}]) for i, t in enumerate(thr_list)])])

# equity curves
f_eq = go.Figure()
for i, s in enumerate(["NCAAF", "NHL", "NBA", "NFL"]):
    thr = 10 if s == "NCAAF" else 6
    d = ev[(ev.sport == s) & (ev.drop_pts >= thr)].sort_values("game_date_et")
    pnl, _ = fade_pnl(d)
    f_eq.add_trace(go.Scatter(x=d.game_date_et, y=pnl.cumsum(), mode="lines", visible=(i == 0),
                              line=dict(color=ACCENT2, width=2.5), name=s,
                              hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f} units<extra></extra>"))
base_layout(f_eq, height=340)
f_eq.add_hline(y=0, line_color=MUTED, line_dash="dash")
f_eq.update_yaxes(title="Cumulative profit (units of $1 contracts)")
f_eq.update_layout(margin=dict(t=50), updatemenus=[dict(
    type="buttons", direction="right", x=0, y=1.18, xanchor="left", bgcolor="white", bordercolor=GRID, font=dict(size=12),
    buttons=[dict(label=lab, method="update", args=[{"visible": [j == i for j in range(4)]}])
             for i, lab in enumerate(["NCAAF, 10+", "NHL, 6+", "NBA, 6+", "NFL, 6+"])])])

# college: edge by drop band
bands = [(0, 3, "0 to 3"), (3, 6, "3 to 6"), (6, 10, "6 to 10"), (10, 101, "10+")]
rows = []
for s in ["NCAAF", "NCAAM"]:
    for lo, hi, lab in bands:
        d = ev[(ev.sport == s) & (ev.drop_pts >= lo) & (ev.drop_pts < hi)]
        if len(d) >= 10:
            rows.append(dict(sport=s, band=lab, **stats(d)))
cb = pd.DataFrame(rows)
f_band = go.Figure()
for s, c in [("NCAAF", ACCENT), ("NCAAM", ACCENT2)]:
    d = cb[cb.sport == s]
    f_band.add_trace(go.Bar(x=d.band, y=d.edge, name=s, marker_color=c,
                            error_y=dict(type="data", symmetric=False, array=d.hi - d.edge, arrayminus=d.edge - d.lo,
                                         color=MUTED, thickness=1.2),
                            customdata=np.stack([d.n, d.win * 100, d.p_start * 100], axis=1),
                            hovertemplate=f"<b>{s}</b>, drop %{{x}} pts<br>%{{customdata[0]:,}} teams"
                                          "<br>Won %{customdata[1]:.1f}% vs priced %{customdata[2]:.1f}%<extra></extra>"))
base_layout(f_band, height=340, barmode="group", showlegend=True,
            legend=dict(orientation="h", y=1.1, x=0))
f_band.update_xaxes(title="Size of pre-game drop (points)")
f_band.update_yaxes(title="Win rate minus price (points)")

# calibration
f_cal = go.Figure()
f_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), hoverinfo="skip", showlegend=False))
for s, c in [("NCAAF", ACCENT), ("NCAAM", ACCENT2)]:
    d = calib[calib.sport == s]
    f_cal.add_trace(go.Scatter(x=d.price, y=d.win, mode="lines+markers", name=s, line=dict(color=c, width=2.5),
                               customdata=d.n, hovertemplate=f"<b>{s}</b><br>Priced %{{x:.1%}}<br>Won %{{y:.1%}}"
                                                             "<br>%{customdata:,} teams<extra></extra>"))
base_layout(f_cal, height=340, showlegend=True, legend=dict(orientation="h", y=1.1, x=0))
f_cal.update_xaxes(title="Kalshi price at start", tickformat=".0%", range=[0, 1])
f_cal.update_yaxes(title="Actual win rate", tickformat=".0%", range=[0, 1])

# Miami at ND
f_mia = go.Figure(go.Scatter(x=mia.t, y=mia.p, mode="lines", line=dict(color=ACCENT, width=2.5),
                             hovertemplate="%{x}<br>Miami %{y:.1%}<extra></extra>"))
news = mia.index[mia.t.str.startswith("Wed Nov 04") & mia.t.str.contains("12 PM")]
if len(news):
    f_mia.add_vline(x=mia.t[news[0]], line_dash="dash", line_color=MUTED)
    f_mia.add_annotation(x=mia.t[news[0]], y=0.47, text="Toney ruled out", showarrow=False, xanchor="left", xshift=6)
base_layout(f_mia, height=320)
f_mia.update_yaxes(tickformat=".0%", range=[0.2, 0.5], title="Miami win price")
f_mia.update_xaxes(nticks=9, title="Hour (ET)")


# star-out games: NFL (injury reports) and college (box-score absences)
def star_row(sport, x, ref_col):
    x = x[x.price_start.between(0.02, 0.98)]
    n = len(x); w = x.won.mean(); pc = x.price_start.mean(); se = np.sqrt((x.price_start * (1 - x.price_start)).sum()) / n
    xr = x[x[ref_col].notna()]
    return dict(sport=sport, n=n, win=w, p_start=pc, edge=(w - pc) * 100, lo=(w - pc - 1.96 * se) * 100,
                hi=(w - pc + 1.96 * se) * 100, p=2 * (1 - norm.cdf(abs((w - pc) / se))),
                move=(xr.price_start - xr[ref_col]).mean() * 100, fade_roi=fade_roi(x))
nfl_clean = nfl[~nfl.both_out.astype(bool)] if "both_out" in nfl else nfl
star = pd.DataFrame([star_row("NFL", nfl_clean, "price_144h"),
                     star_row("NCAAF", col[col.sport == "NCAAF"], "price_ref"),
                     star_row("NCAAM", col[col.sport == "NCAAM"], "price_ref")])
_sx = pd.concat([nfl_clean.rename(columns={"price_144h": "price_ref"})[["price_ref", "price_start", "won"]],
                 col[["price_ref", "price_start", "won"]]])
_sx = _sx[_sx.price_start.between(0.02, 0.98)]
star_tot = dict(n=len(_sx), win=_sx.won.mean(), p_start=_sx.price_start.mean(),
                move=(_sx.dropna(subset=["price_ref"]).price_start - _sx.dropna(subset=["price_ref"]).price_ref).mean() * 100,
                fade_roi=fade_roi(_sx), back_roi=None)
_a = (_sx.price_start + 0.01).clip(upper=0.99); _f = 0.07 * _a * (1 - _a)
star_tot["back_roi"] = ((_sx.won - _a - _f).sum() / (_a + _f).sum())
f_star = dot_ci(star, "edge", "lo", "hi", 0, lambda v: f"{v:+.1f}", "Win rate minus price at start (points)",
                label=lambda r: f"{r.sport} ({r.n:,})")

# ---------------------------------------------------------------- tables
def table(df, cols, fmts, cls=""):
    th = "".join(f"<th>{html.escape(c)}</th>" for c in cols.values())
    body = []
    for _, r in df.iterrows():
        tds = []
        for k in cols:
            v = r[k]
            txt = fmts.get(k, str)(v) if pd.notna(v) else "n/a"
            num = isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
            tds.append(f'<td class="{"n" if num else ""}">{html.escape(str(txt))}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


pct = lambda v: f"{v:.1%}"
bt_tab = bt.copy()
bt_tab["rule"] = bt_tab.apply(lambda r: f"{r.sport}, drop {int(r.thr)}+", axis=1)
t_bt = table(bt_tab.sort_values("roi", ascending=False),
             {"rule": "Rule", "n": "Bets", "roi": "ROI", "first": "First 60%", "last": "Last 40%"},
             {"n": lambda v: f"{int(v):,}", "roi": lambda v: f"{v:+.1%}", "first": lambda v: f"{v:+.1%}", "last": lambda v: f"{v:+.1%}"})
nfl_t = nfl.assign(move=(nfl.price_start - nfl.price_144h) * 100, res=np.where(nfl.won == 1, "Won", "Lost"))
t_nfl = table(nfl_t.sort_values("move"),
              {"game_id": "Game", "team": "Team", "players": "Star out", "tier": "Tier", "price_144h": "6 days before",
               "price_start": "At start", "move": "Move (pts)", "res": "Result", "score": "Score"},
              {"price_144h": pct, "price_start": pct, "move": lambda v: f"{v:+.1f}"})
col_t = col.assign(res=np.where(col.won == 1, "Won", "Lost"), date=col.date.astype(str))
t_col = table(col_t.sort_values("move_pts"), {"date": "Date", "sport": "Sport", "team": "Team", "opp": "Opponent",
              "players": "Star out", "star_profile": "Profile", "price_ref": "Before", "price_start": "At start", "move_pts": "Move (pts)", "res": "Result"},
              {"price_ref": pct, "price_start": pct, "move_pts": lambda v: f"{v:+.1f}"}, cls="search2")
nhl = bt[(bt.sport == "NHL") & (bt.thr == 6)].iloc[0]
cf = bt[(bt.sport == "NCAAF") & (bt.thr == 10)].iloc[0]
nfl_bt = bt[(bt.sport == "NFL") & (bt.thr == 6)].iloc[0]
nba_bt = bt[(bt.sport == "NBA") & (bt.thr == 6)].iloc[0]
ncaaf, ncaam = by_sport.set_index("sport").loc["NCAAF"], by_sport.set_index("sport").loc["NCAAM"]
nhl_s = by_sport.set_index("sport").loc["NHL"]

# ---------------------------------------------------------------- page
page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Star Injury Overreaction Study</title>
<meta name="description" content="Do prediction markets overreact to star injury news? A backtest on {len(ev):,} Kalshi game markets.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --ink:{INK}; --muted:#5b6675; --line:#dfe3e8; --bg:#f7f6f2; --panel:#ffffff; --accent:{ACCENT}; --accent2:{ACCENT2}; --good:{GOOD}; --bad:{BAD}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 "IBM Plex Sans", system-ui, sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:48px max(20px, env(safe-area-inset-left)) 80px; }}
header .kicker {{ font:500 12px "IBM Plex Mono", monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); }}
h1 {{ font:600 clamp(32px,5vw,52px)/1.08 Fraunces, Georgia, serif; margin:10px 0 16px; text-wrap:balance; }}
h2 {{ font:600 26px/1.2 Fraunces, Georgia, serif; margin:64px 0 8px; }}
h3 {{ font-size:16px; font-weight:600; margin:0 0 4px; }}
p.lede {{ font-size:19px; max-width:64ch; color:#2c3a4b; }}
p.note, .card p {{ color:var(--muted); font-size:14px; margin:0 0 12px; max-width:70ch; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:32px 0 8px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
.tile .v {{ font:400 34px Fraunces, Georgia, serif; font-variant-numeric:tabular-nums; }}
.tile .l {{ font-size:13px; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); gap:16px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:20px; min-width:0; overflow:hidden; }}
.card .plotly-graph-div {{ width:100% !important; }}
.verdict {{ border-left:4px solid var(--accent2); background:var(--panel); padding:16px 20px; border-radius:0 10px 10px 0; margin:16px 0; }}
.tbox {{ max-height:440px; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th {{ position:sticky; top:0; background:#f1f3f5; text-align:left; font-weight:600; padding:8px 10px; border-bottom:1px solid var(--line); }}
td {{ padding:7px 10px; border-bottom:1px solid #eef0f3; }}
td.n {{ text-align:right; }}
.tbox td, .tbox th {{ white-space:nowrap; }}
input[type=search] {{ font:inherit; font-size:14px; padding:8px 12px; border:1px solid var(--line); border-radius:8px; width:min(320px,100%); margin-bottom:10px; }}
dl {{ display:grid; grid-template-columns:minmax(140px,220px) 1fr; gap:10px 20px; font-size:14.5px; }}
dt {{ font-weight:600; }} dd {{ margin:0; color:#334155; }}
code {{ font-family:"IBM Plex Mono", monospace; font-size:13px; background:#eef0f3; padding:1px 5px; border-radius:4px; }}
footer {{ margin-top:72px; font-size:13px; color:var(--muted); border-top:1px solid var(--line); padding-top:16px; }}
@media (max-width:640px) {{ dl {{ grid-template-columns:1fr; }} .wrap {{ padding-top:28px; }} }}
</style></head>
<body><div class="wrap">

<header>
  <div class="kicker">Kalshi game markets, 2025 to Nov 2026</div>
  <h1>Do markets overreact when a star gets hurt?</h1>
  <p class="lede">No. In {star_tot['n']:,} games where a statistically defined star did not play, the injured team's Kalshi price fell
  only {abs(star_tot['move']):.1f} points on average, and those teams won {star_tot['win']:.1%} of the time against {star_tot['p_start']:.1%}
  implied by their final price. The market does not overreact: backing the injured team returns {star_tot['back_roi']:+.1%} after fees,
  and betting against it roughly breaks even ({star_tot['fade_roi']:+.1%}).</p>
</header>

<div class="tiles">
  <div class="tile"><div class="v">{star_tot['n']:,}</div><div class="l">Games with a star out (NFL, NCAAF, NCAAM)</div></div>
  <div class="tile"><div class="v">{star_tot['move']:+.1f}</div><div class="l">Avg price move, points</div></div>
  <div class="tile"><div class="v">{star_tot['win']:.1%}</div><div class="l">How often the injured team won</div></div>
  <div class="tile"><div class="v">{star_tot['p_start']:.1%}</div><div class="l">Implied by price at start</div></div>
  <div class="tile"><div class="v" style="color:var(--bad)">{star_tot['back_roi']:+.1%}</div><div class="l">ROI backing the injured team</div></div>
</div>

<h2>Star players ruled out</h2>
<p class="note">Only games where a star is confirmed missing: NFL injury reports, and college box scores with no line for the star.
Right of zero would mean the market overreacted (the original thesis). Lines are 95% intervals.</p>
<div class="grid">
  <div class="card"><h3>Injured team: win rate minus price at start</h3><p>Games where both teams were missing a star are left out.</p>{html_fig(f_star, "star")}</div>
  <div class="card"><h3>NFL star-out games</h3><p>QB or skill star listed Out or Doubtful with no stat line.</p><div class="tbox">{t_nfl}</div></div>
</div>
<div class="card" style="margin-top:16px"><h3>College star-out games, biggest price moves first</h3>
<p>NCAAF and NCAAM stars with no box-score line in a game their team played.</p>
<input type="search" id="q2" placeholder="Filter by team, player or sport" aria-label="Filter college table"><div class="tbox">{t_col}</div></div>

<h2>The motivating case: Miami at Notre Dame</h2>
<div class="card"><h3>Kalshi price for Miami to win, hourly</h3>
<p>Malachi Toney was ruled out Wednesday, Nov 4. Notre Dame won 31 to 24, a result well inside what the 29.5% closing price allowed.</p>
{html_fig(f_mia, "mia")}</div>

<h2>All pre-game price moves, for any reason</h2>
<p class="note">A broader check that does not depend on injury data: every game where a team's price fell before the start, whatever the
cause (injuries, lineups, rest, goalie or pitcher news, or traders correcting an early price). Across {len(ev):,} games with a traded price
before the news, the {head['n']:,} teams that fell 6+ points won {head['win']:.1%} against {head['p_start']:.1%} priced.</p>
<div class="grid">
  <div class="card"><h3>Teams that fell 6+ points</h3><p>Price before = 6 days out for NFL and NCAAF, 72 hours out elsewhere.</p>{html_fig(f_edge, "edge")}</div>
  <div class="card"><h3>Drops in the final 24 hours only</h3><p>When injury and lineup news usually lands. Drops of 5+ points.</p>{html_fig(f_late, "late")}</div>
</div>

<h2>Backtest: bet against teams after a price drop</h2>
<p class="note">Rule: when a team's Kalshi price falls by the threshold, buy its opponent in the last hour before the start,
paying 1 cent over the midpoint plus Kalshi's taker fee (0.07 × P × (1 minus P)). The first 60% of games and the last 40% are shown
separately as a holdout check.</p>
<div class="grid">
  <div class="card"><h3>ROI by sport</h3>{html_fig(f_bt, "bt")}</div>
  <div class="card"><h3>Cumulative profit, in date order</h3>{html_fig(f_eq, "eq")}</div>
</div>
<div class="verdict"><b>Verdict.</b> No rule clears the bar. Fading NCAAF teams after a 10+ point drop returns {cf.roi:+.1%}
({int(cf.n):,} bets), and NHL at 6+ returns {nhl.roi:+.1%}. NBA at 6+ about breaks even ({nba_bt.roi:+.1%} on {int(nba_bt.n)} bets), and
NFL at 6+ shows {nfl_bt.roi:+.1%} on just {int(nfl_bt.n)} bets, too few to trust. Fading the injured team in star-out games returns
{star_tot['fade_roi']:+.1%}. The closing price already reflects the news. This is research, not financial advice.</div>
<div class="card"><h3>Every rule tested</h3><div class="tbox">{t_bt}</div></div>

<h2>College price moves in detail</h2>
<p class="note">NCAAF: {int(ncaaf.n):,} teams that fell 6+ won {ncaaf.win:.1%} vs {ncaaf.p_start:.1%} priced.
NCAAM: {int(ncaam.n):,} teams won {ncaam.win:.1%} vs {ncaam.p_start:.1%}. College prices hold up as well as the pros.</p>
<div class="grid">
  <div class="card"><h3>Edge by size of drop</h3><p>Bars show 95% intervals.</p>{html_fig(f_band, "band")}</div>
  <div class="card"><h3>Are college closing prices calibrated?</h3><p>Points on the dashed line mean prices matched results.</p>{html_fig(f_cal, "cal")}</div>
</div>

<h2>Definitions and method</h2>
<div class="card"><dl>
<dt>Star player (NFL)</dt><dd>QB: threw at least 60% of the team's pass attempts in that season's prior games, with at least 3 prior games.
Skill player: the WR, TE or RB who leads the team in scrimmage yards per game in prior games, averaging at least 70, with at least 3 prior games.
Fixed before looking at results.</dd>
<dt>Star player (NCAAF)</dt><dd>Same rule as the NFL from ESPN box scores: the QB with 60%+ of pass attempts, or the non-passer leading
the team in scrimmage yards per game at 70+, each over 3+ prior games. Out = no box-score line in a game his team played, after appearing in one of the team's previous 3 games.</dd>
<dt>Star player (NCAAM)</dt><dd>In his team's top two for minutes per game and averaging 12+ points, over 5+ prior games played (and at least half the team's games).
Out = no minutes in a game his team played, after playing in one of the team's previous 5 games. A box score confirms the absence, not always its cause.</dd>
<dt>Star-out event (NFL)</dt><dd>The star is listed Out or Doubtful on the final weekly injury report and has no stat line that week.</dd>
<dt>Price drop (all sports)</dt><dd>The side whose Kalshi price fell between the reference time and the start. Not limited to injuries: it captures any other
pre-game news (lineups, rest, starting goalie or pitcher).</dd>
<dt>Price before the news</dt><dd>6 days before start for NFL and NCAAF, 72 hours for other sports. It must rest on at least 100 contracts traded; if not,
the first hour that reaches 100 contracts is used (at least 12 hours before start), otherwise the game is left out. This removes stale opening
prices in thin markets, such as a lone 1-contract trade at 97%.</dd>
<dt>Price</dt><dd>Midpoint of best bid and ask at the end of each hour (last trade if the spread is wider than 10 cents). Start time is Kalshi's
occurrence time minus 3 hours, confirmed by the price jump at the start of play.</dd>
<dt>Edge</dt><dd>Actual win rate minus average price at start. Intervals use the variance implied by the prices.</dd>
<dt>Coverage</dt><dd>Kalshi single-game markets begin with each league's 2025 season, so 2023 and 2024 are not included. Preseason games and prices under 2
or over 98 cents are excluded.</dd>
<dt>Data</dt><dd>Kalshi public API (<code>trade-api/v2</code> markets, historical markets, hourly candlesticks), pulled Nov 8 2026. NFL injuries and
sportsbook lines from nflverse. College box scores from ESPN via the sportsdataverse project.</dd>
</dl></div>

<footer>Last built {pd.Timestamp.now(tz="America/New_York").strftime("%b %d, %Y at %I:%M %p ET")} from {len(ev):,} games. Built with Python and Plotly from the CSVs in <code>data/</code>. Rebuild with <code>python build_site.py</code>.</footer>
</div>
<script>
const fit = () => document.querySelectorAll('.plotly-graph-div').forEach(d => window.Plotly && Plotly.Plots.resize(d));
window.addEventListener('load', fit);
window.addEventListener('resize', fit);
const q2 = document.getElementById('q2');
q2 && q2.addEventListener('input', () => {{
  const s = q2.value.toLowerCase();
  document.querySelectorAll('table.search2 tbody tr').forEach(tr => tr.hidden = !tr.textContent.toLowerCase().includes(s));
}});
</script>
</body></html>"""

OUT.mkdir(exist_ok=True)
(OUT / "index.html").write_text(page, encoding="utf-8")
print(f"wrote {OUT / 'index.html'} ({len(page) / 1e3:.0f} KB)")
