"""Kalshi pre-game move study: do markets overreact to pre-game news?

Input: kalshi_export.json  {M: [[ticker, event, yes_team, result, occ, open, vol, close, series], ...],
                            F: [{tk, K, m:[mid*1000 for h=-192..+6], v:[vol]}, ...]}
K = assumed start time = occurrence_datetime - 3h.
"""
import json, numpy as np, pandas as pd
from scipy.stats import norm, binomtest

SPORT = {'KXNFLGAME': 'NFL', 'KXNCAAFGAME': 'NCAAF', 'KXNCAAMBGAME': 'NCAAM',
         'KXNBAGAME': 'NBA', 'KXNHLGAME': 'NHL', 'KXMLBGAME': 'MLB'}
H0 = 192  # index of hour 0 (= K)

d = json.load(open('kalshi_export.json'))
M = pd.DataFrame(d['M'], columns=['ticker', 'event', 'yes_team', 'result', 'occ', 'open', 'vol', 'close', 'series'])
M = M.drop_duplicates('ticker')
F = {f['tk']: f for f in d['F']}
print('markets', len(M), 'features', len(F))

# team names per event (both sides)
names = M.groupby('event').yes_team.apply(list).to_dict()

rows = []
for r in M[M.ticker.isin(F.keys())].itertuples():
    f = F[r.ticker]
    m = np.array([np.nan if x is None else x / 1000 for x in f['m']])
    v = np.array(f['v'], dtype=float)
    other = [n for n in names.get(r.event, []) if n != r.yes_team]
    rows.append(dict(ticker=r.ticker, event=r.event, sport=SPORT[r.series], team=r.yes_team,
                     opp=other[0] if other else None, won=int(r.result == 'yes'),
                     K=pd.to_datetime(f['K'], unit='s', utc=True), vol=r.vol,
                     m=m, v=v))
G = pd.DataFrame(rows)

# Kickoff check: hour of peak volume relative to K within [-6, +6]
peak = G.v.apply(lambda v: int(np.argmax(v[H0 - 6:H0 + 7])) - 6)
print('peak-volume hour relative to assumed start, by sport:')
print(pd.crosstab(G.sport, peak.clip(-3, 4)))


def at(m, h):
    """mid price h hours before start (h>=0)."""
    return m[H0 - h]


G['p_close'] = G.m.apply(lambda m: at(m, 0))
G['p_1h'] = G.m.apply(lambda m: at(m, 1))
for h in (6, 24, 48, 72, 120, 144, 168):
    G[f'p_{h}'] = G.m.apply(lambda m, h=h: at(m, h))
# earliest available price inside 7 days (first non-null)
G['p_first'] = G.m.apply(lambda m: next((x for x in m[:H0] if not np.isnan(x)), np.nan))
G['hrs_first'] = G.m.apply(lambda m: H0 - next((i for i, x in enumerate(m[:H0]) if not np.isnan(x)), H0))
G['pre_vol'] = G.v.apply(lambda v: v[:H0].sum())
G['season'] = G.K.dt.year.where(G.K.dt.month >= 7, G.K.dt.year - 1)
# Miami at Notre Dame price path (Miami perspective)
mi = G[G.event == 'KXNCAAFGAME-26NOV07MIAND']
if len(mi):
    r = mi.iloc[0]
    flip = r.team != 'Miami (FL)' and 'Miami' not in str(r.team)
    t0 = (r.K.floor('h'))
    pts = []
    for i, (p, v) in enumerate(zip(r.m, r.v)):
        h = i - H0
        if h > 0:
            break
        ts = (t0 + pd.Timedelta(hours=h)).tz_convert('America/New_York').strftime('%a %b %d %I %p')
        pts.append(dict(t=ts, p=None if np.isnan(p) else round(1 - p if flip else p, 3), v=int(v)))
    pd.DataFrame(pts).dropna().to_pickle('miami_path.pkl')
    print('Miami path saved; stored side', r.team)
G = G.drop(columns=['m', 'v'])
G.to_pickle('games.pkl')
print(G.groupby('sport').agg(n=('ticker', 'size'), first=('K', 'min'), last=('K', 'max'),
                             has_close=('p_close', lambda x: x.notna().mean())))
