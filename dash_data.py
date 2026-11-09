"""CSV datasets for the dashboard."""
import numpy as np, pandas as pd
E = pd.read_pickle('events.pkl'); T = pd.read_pickle('tables.pkl')
ORDER = ['NCAAF', 'NCAAM', 'NFL', 'NBA', 'NHL', 'MLB']
rng = np.random.default_rng(7)
big = E[E['drop'] >= 6]


def stats(df):
    n = len(df); w, pc, pr = df.won.mean(), df.p_close.mean(), df.p_ref.mean()
    se = np.sqrt((df.p_close * (1 - df.p_close)).sum()) / n
    from scipy.stats import norm
    z = (w - pc) / se
    kept = (w - pr) / (pc - pr)
    bs = []
    a = df[['won', 'p_ref', 'p_close']].to_numpy()
    for _ in range(1000):
        s = a[rng.integers(0, n, n)]
        bs.append((s[:, 0].mean() - s[:, 1].mean()) / (s[:, 2].mean() - s[:, 1].mean()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=n, avg_drop=df['drop'].mean(), p_ref=pr, p_close=pc, win_rate=w, edge_pts=(w - pc) * 100,
                ci_lo=(w - pc - 1.96 * se) * 100, ci_hi=(w - pc + 1.96 * se) * 100, p_value=2 * (1 - norm.cdf(abs(z))),
                move_kept=kept, kept_lo=lo, kept_hi=hi, roi=df.roi.mean())


bs = pd.DataFrame([dict(sport=s, **stats(big[big.sport == s])) for s in ORDER if (big.sport == s).any()])
bs.round(4).to_csv('dash/by_sport.csv', index=False)
bb = []
for (s, b), df in E[E.sport.isin(['NCAAF', 'NCAAM'])].groupby(['sport', 'bucket'], observed=True):
    if len(df) >= 10:
        bb.append(dict(sport=s, bucket=str(b), **stats(df)))
pd.DataFrame(bb).round(4).to_csv('dash/by_bucket.csv', index=False)
c = T['calib']; c[c.sport.isin(['NCAAF', 'NCAAM'])].round(4).to_csv('dash/calib.csv', index=False)
h = dict(games=len(E), big_moves=len(big), win_rate=big.won.mean(), p_close=big.p_close.mean(),
         p_ref=big.p_ref.mean(), roi=big.roi.mean())
pd.DataFrame([h]).round(4).to_csv('dash/headline.csv', index=False)
m = pd.read_pickle('miami_path.pkl').reset_index(drop=True); m['i'] = range(len(m))
m[['i', 't', 'p']].to_csv('dash/miami.csv', index=False)
ns = T['NS'][['game_id', 'team', 'opp', 'players', 'tier', 'p_144', 'p_close', 'move_pts', 'won', 'score']].dropna(subset=['p_close'])
ns.round(4).to_csv('dash/nfl_star.csv', index=False)
ev = E[E['drop'] >= 10].copy()
ev['date'] = ev.K.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
ev[['ticker', 'date', 'sport', 'team', 'opp', 'p_ref', 'p_close', 'drop', 'won']].round(4).to_csv('dash/events.csv', index=False)
print(bs.round(3).to_string()); print(h)
import os; print({f: os.path.getsize('dash/' + f) for f in os.listdir('dash')})
L = pd.read_pickle('late24.pkl'); L5 = L[L['drop'] >= 5]
lt = pd.DataFrame([dict(sport=s, **stats(L5[L5.sport == s])) for s in ORDER if (L5.sport == s).sum() >= 10])
lt.round(4).to_csv('dash/late24.csv', index=False); print(lt.round(3).to_string())

# ---- star-out games: NFL (nflverse injury reports) + NCAAF / NCAAM (box-score absences)
from scipy.stats import norm as _norm
C = pd.read_pickle('college_stars.pkl')
C = C[~C.both_out & C.p_close.between(0.02, 0.98)].copy()
NF = T['NS'].copy(); NF = NF[~NF.both_out & NF.p_close.between(0.02, 0.98)]
NF = NF.assign(sport='NFL', p_ref=NF.p_144, players=NF.players, date=NF.game_id)
def star_stats(x):
    n = len(x); w = x.won.mean(); pc = x.p_close.mean(); se = np.sqrt((x.p_close * (1 - x.p_close)).sum()) / n
    a = ((1 - x.p_close) + 0.01).clip(upper=0.99); f = 0.07 * a * (1 - a)
    xr = x[x.p_ref.notna()]
    return dict(n=n, move_pts=(xr.p_close - xr.p_ref).mean() * 100, p_ref=xr.p_ref.mean(), p_close=pc, win_rate=w,
                edge_pts=(w - pc) * 100, ci_lo=(w - pc - 1.96 * se) * 100, ci_hi=(w - pc + 1.96 * se) * 100,
                p_value=2 * (1 - _norm.cdf(abs((w - pc) / se))), fade_roi=((1 - x.won) - a - f).sum() / (a + f).sum())
ss = [dict(sport='NFL', **star_stats(NF)), dict(sport='NCAAF', **star_stats(C[C.sport == 'NCAAF'])),
      dict(sport='NCAAM', **star_stats(C[C.sport == 'NCAAM']))]
pd.DataFrame(ss).round(4).to_csv('dash/star_summary.csv', index=False)
cs = C.assign(move_pts=(C.p_close - C.p_ref) * 100)[['ticker', 'sport', 'date', 'team', 'opp', 'players', 'tier', 'first_game_out',
                                                      'p_ref', 'p_close', 'move_pts', 'won']]
cs.sort_values(['sport', 'date']).round(4).to_csv('dash/college_star.csv', index=False)
print(pd.DataFrame(ss).round(3).to_string()); print('college star rows', len(cs))
