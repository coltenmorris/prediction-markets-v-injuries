"""Main tests on games.pkl (built by analyze.py)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm

G = pd.read_pickle('games.pkl')
G = G[G.p_close.between(0.02, 0.98)].copy()

# drop preseason / exhibition windows
mo, dy = G.K.dt.month, G.K.dt.day
pre = (((G.sport == 'NBA') & (mo.isin([7, 8, 9]) | ((mo == 10) & (dy < 20)))) |
       ((G.sport == 'NHL') & (mo.isin([7, 8, 9]) | ((mo == 10) & (dy < 6)))) |
       ((G.sport == 'MLB') & ((mo < 3) | ((mo == 3) & (dy < 25)))) |
       ((G.sport == 'NFL') & ((mo == 8) | ((mo == 9) & (dy < 4)))))
G = G[~pre].copy()

# reference (pre-news) price. Weekly sports (NFL, NCAAF): 144h before start, i.e. right after the
# previous week's games, before the week's injury news. Others: 72h, else earliest price >= 12h before.
weekly = G.sport.isin(['NFL', 'NCAAF'])
G['p_ref'] = np.where(weekly & G.p_144.notna(), G.p_144,
              np.where(G.p_72.notna(), G.p_72, np.where(G.hrs_first >= 12, G.p_first, np.nan)))
G['ref_h'] = np.where(weekly & G.p_144.notna(), 144,
              np.where(G.p_72.notna(), 72, np.where(G.hrs_first >= 12, G.hrs_first, np.nan)))
G = G[G.p_ref.between(0.02, 0.98)].copy()

# orient every game to the team the market moved AGAINST (the "marked down" team)
down = G.p_close < G.p_ref
E = pd.DataFrame({
    'sport': G.sport, 'season': G.season, 'K': G.K, 'event': G.event, 'ticker': G.ticker,
    'team': np.where(down, G.team, G.opp), 'opp': np.where(down, G.opp, G.team),
    'p_ref': np.where(down, G.p_ref, 1 - G.p_ref),
    'p_24': np.where(down, G.p_24, 1 - G.p_24),
    'p_close': np.where(down, G.p_close, 1 - G.p_close),
    'won': np.where(down, G.won, 1 - G.won),
    'ref_h': G.ref_h, 'pre_vol': G.pre_vol, 'vol': G.vol,
})
E['drop'] = ((E.p_ref - E.p_close) * 100).round(6)           # points the team was marked down (>=0)
E['late_share'] = ((E.p_24 - E.p_close) / (E.p_ref - E.p_close)).where(E['drop'] > 0)
E['fav'] = np.where(E.p_close >= 0.5, 'Favorite at close', 'Underdog at close')
E['bucket'] = pd.cut(E['drop'], [-0.01, 3, 6, 10, 100], labels=['0-3 pts', '3-6 pts', '6-10 pts', '10+ pts'])
ask = (E.p_close + 0.01).clip(upper=0.99)
fee = 0.07 * ask * (1 - ask)
E['roi'] = (E.won - ask - fee) / (ask + fee)     # buy the marked-down team at close, taker fee
E.to_pickle('events.pkl')


def summ(df):
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)
    w, pc, pr = df.won.mean(), df.p_close.mean(), df.p_ref.mean()
    se = np.sqrt((df.p_close * (1 - df.p_close)).sum()) / n
    z = (w - pc) / se
    return pd.Series({'n': n, 'avg_drop_pts': df['drop'].mean(), 'p_ref': pr, 'p_close': pc, 'win_rate': w,
                      'edge_vs_close_pts': (w - pc) * 100, 'z': z, 'p_value': 2 * (1 - norm.cdf(abs(z))),
                      'move_kept': (w - pr) / (pc - pr) if abs(pc - pr) > 1e-9 else np.nan,
                      'brier_ref': ((df.won - df.p_ref) ** 2).mean(), 'brier_close': ((df.won - df.p_close) ** 2).mean(),
                      'roi_pct': df.roi.mean() * 100})


def logit(p):
    return np.log(p / (1 - p))


def reg(df):
    X = pd.DataFrame({'logit_close': logit(df.p_close), 'move': logit(df.p_close) - logit(df.p_ref)})
    X = sm.add_constant(X)
    r = sm.Logit(df.won, X).fit(disp=0)
    return pd.Series({'b_close': r.params.logit_close, 'b_move': r.params.move, 'se_move': r.bse.move,
                      'p_move': r.pvalues.move, 'n': len(df)})


if __name__ == '__main__':
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
    big = E[E['drop'] >= 6]
    print('=== Marked down >= 6 pts, by sport ===')
    print(big.groupby('sport').apply(summ).round(3))
    print('\n=== By sport x bucket ===')
    print(E.groupby(['sport', 'bucket'], observed=True).apply(summ).round(3))
    print('\n=== Regression: won ~ logit(close) + move (move<0 coef => overreaction) ===')
    print(E.groupby('sport').apply(reg).round(3))
    print('\n=== >=6 pts, fav vs dog ===')
    print(big.groupby(['sport', 'fav']).apply(summ).round(3))

# ---- late-news cut: moves inside the final 24 hours (when injury and lineup news usually lands) ----
G24 = G[G.p_24.between(0.02, 0.98)].copy()
d24 = G24.p_close < G24.p_24
L = pd.DataFrame({'sport': G24.sport, 'season': G24.season, 'K': G24.K, 'ticker': G24.ticker,
                  'team': np.where(d24, G24.team, G24.opp), 'opp': np.where(d24, G24.opp, G24.team),
                  'p_ref': np.where(d24, G24.p_24, 1 - G24.p_24),
                  'p_close': np.where(d24, G24.p_close, 1 - G24.p_close),
                  'won': np.where(d24, G24.won, 1 - G24.won)})
L['drop'] = ((L.p_ref - L.p_close) * 100).round(6)
_a = (L.p_close + 0.01).clip(upper=0.99); _f = 0.07 * _a * (1 - _a)
L['roi'] = (L.won - _a - _f) / (_a + _f)
L.to_pickle('late24.pkl')
