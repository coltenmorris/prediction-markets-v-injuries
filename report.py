"""Build summary tables, NFL star-out tagging, event spreadsheet data and dashboard data."""
import json, numpy as np, pandas as pd
from study import E, summ, reg

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
ORDER = ['NCAAF', 'NCAAM', 'NFL', 'NBA', 'NHL', 'MLB']

# ---------- NFL star-out tagging ----------
nfl = pd.read_csv('/mnt/user-data/uploads/nfl_star_injury_events.csv')
nfl = nfl[nfl.season >= 2025].copy()
fix = {'JAC': 'JAX', 'LAR': 'LA'}
G = pd.read_pickle('games.pkl')
ng = G[G.sport == 'NFL'].copy()
code = ng.event.str.split('-').str[1]                         # e.g. 26NOV01PHIWAS
ng['date'] = pd.to_datetime(code.str[:7], format='%y%b%d').dt.strftime('%Y-%m-%d')
ng['tm'] = ng.ticker.str.split('-').str[-1].replace(fix)
games = pd.read_csv('raw/games.csv')
games = games[games.season >= 2025]
key = {}
for g in games.itertuples():
    key[(g.gameday, g.home_team)] = g.game_id
    key[(g.gameday, g.away_team)] = g.game_id
ng['game_id'] = [key.get((d, t)) for d, t in zip(ng.date, ng.tm)]
# team-level Kalshi row for each nflverse (game, team)
H0 = None
rows = []
for r in nfl.itertuples():
    m = ng[(ng.game_id == r.game_id)]
    if m.empty:
        continue
    m = m.iloc[0]
    same = m.tm == r.team
    p = lambda x: x if same else 1 - x
    rows.append(dict(game_id=r.game_id, season=r.season, week=r.week, team=r.team, opp=r.opp,
                     players=r.players, tier=r.tier, first_game_out=r.first_game_out,
                     both_out=r.both_teams_missing_star, book_spread=r.team_spread,
                     p_144=p(m.p_144), p_72=p(m.p_72), p_24=p(m.p_24), p_close=p(m.p_close),
                     won=m.won if same else 1 - m.won, score=r.score_team_opp, covered=r.covered))
NS = pd.DataFrame(rows)
NS['move_pts'] = (NS.p_close - NS.p_144) * 100
print('NFL star-out games matched to Kalshi:', len(NS), 'of', len(nfl))
print(NS[['game_id', 'team', 'players', 'tier', 'p_144', 'p_close', 'move_pts', 'won']].round(3).to_string())
nclean = NS[~NS.both_out]
print('\nNFL star out (clean): n=%d avg move %.1f pts, p_144 %.3f, p_close %.3f, win %.3f' % (
    len(nclean), nclean.move_pts.mean(), nclean.p_144.mean(), nclean.p_close.mean(), nclean.won.mean()))

# ---------- main tables ----------
big = E[E['drop'] >= 6]
by_sport = big.groupby('sport').apply(summ).reindex(ORDER)
by_bucket = E.groupby(['sport', 'bucket'], observed=True).apply(summ)
regs = E.groupby('sport').apply(reg).reindex(ORDER)
favdog = big.groupby(['sport', 'fav']).apply(summ)
# late moves (most of the drop in the final 24h) vs early
big2 = big.assign(timing=np.where(big.late_share >= 0.5, 'Mostly final 24h', 'Mostly earlier'))
timing = big2.groupby(['sport', 'timing']).apply(summ)
# liquidity terciles within sport
E['liq'] = E.groupby('sport').pre_vol.transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=['Thin', 'Mid', 'Deep']))
liq = E[E['drop'] >= 6].groupby(['sport', 'liq'], observed=True).apply(summ)
season = big.groupby(['sport', 'season']).apply(summ)
print('\n== >=6 pts by sport ==\n', by_sport.round(3))
print('\n== regression ==\n', regs.round(3))
print('\n== timing ==\n', timing.round(3))
print('\n== liquidity ==\n', liq.round(3))
print('\n== season ==\n', season.round(3))

# all-games calibration of closing price (baseline efficiency) per sport
G2 = pd.read_pickle('games.pkl')
G2 = G2[G2.ticker.isin(E.ticker)]
calib = []
for s, df in G2.groupby('sport'):
    df = df[df.p_close.between(0.02, 0.98)]
    b = pd.cut(df.p_close, np.arange(0, 1.01, 0.1))
    for k, d in df.groupby(b, observed=True):
        calib.append(dict(sport=s, bin=str(k), mid=d.p_close.mean(), win=d.won.mean(), n=len(d)))
calib = pd.DataFrame(calib)

pd.to_pickle(dict(by_sport=by_sport, by_bucket=by_bucket, regs=regs, favdog=favdog, timing=timing,
                  liq=liq, season=season, calib=calib, NS=NS), 'tables.pkl')
