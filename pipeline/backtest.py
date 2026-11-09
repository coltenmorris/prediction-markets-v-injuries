import numpy as np, pandas as pd
from scipy.stats import norm
E = pd.read_pickle('events.pkl').sort_values('K')
def run(df, side='fade', fee=0.07, slip=0.01):
    # fade = buy the OTHER team (the one the market moved toward) at start
    p = (1 - df.p_close) if side == 'fade' else df.p_close
    y = (1 - df.won) if side == 'fade' else df.won
    a = (p + slip).clip(upper=0.99); f = fee * a * (1 - a)
    pnl = y - a - f
    return pd.Series({'n': len(df), 'price_paid': a.mean(), 'win': y.mean(), 'roi_pct': pnl.sum() / (a + f).sum() * 100,
                      'units': pnl.sum(), 'se_roi': pnl.std() / np.sqrt(len(df)) / (a + f).mean() * 100 if len(df) > 1 else np.nan})
rows = []
for sp in ['NFL', 'NBA', 'NHL', 'NCAAF', 'NCAAM', 'MLB', 'NBA+NHL+NFL']:
    d0 = E[E.sport.isin(['NBA', 'NHL', 'NFL'])] if '+' in sp else E[E.sport == sp]
    for thr in [6, 10, 15]:
        d = d0[d0['drop'] >= thr]
        if len(d) < 15: continue
        cut = d.K.quantile(0.6)
        for lab, dd in [('all', d), ('first 60%', d[d.K <= cut]), ('last 40%', d[d.K > cut])]:
            for fee, slip, ex in [(0.07, 0.01, 'taker'), (0.0175, 0.0, 'maker at mid')]:
                r = run(dd, 'fade', fee, slip); rows.append(dict(sport=sp, thr=thr, sample=lab, exec=ex, **r))
R = pd.DataFrame(rows)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300)
print(R.round(3).to_string())
