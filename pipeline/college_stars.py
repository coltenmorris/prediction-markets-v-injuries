"""College star-out events from ESPN box scores (sportsdataverse), matched to Kalshi prices.

NCAAM star: in his team's top two for minutes per game AND averaging 12+ points, over at least 5 prior games played
            this season (and at least half the team's prior games). Out = his team played and he has no box-score
            minutes, while he played in at least one of the team's previous 5 games.
NCAAF star: QB = 60%+ of the team's pass attempts in prior games; skill = the team's leader in scrimmage yards per game
            (rushing + receiving) at 70+, excluding passers; both need 3+ prior games. Out = no box-score line in a game
            the team played, while he appeared in at least one of the team's previous 3 games.
"""
import re, numpy as np, pandas as pd
from rapidfuzz import fuzz

CB = 'cbx/'
H0 = 192


# ------------------------------------------------------------------ NCAAM
def ncaam_events():
    b = pd.concat([pd.read_parquet(CB + f'espn_mens_college_basketball_player_boxscores_player_box_{y}.parquet') for y in (2026, 2027)])
    b = b[b.athlete_id.notna()].copy()
    b['played'] = (~b.did_not_play.fillna(False).astype(bool)) & (b.minutes.fillna(0) > 0)
    b['game_date'] = b.game_date.astype(str)
    tg = (b.groupby(['season', 'team_id', 'game_id']).agg(date=('game_date', 'first'), team=('team_location', 'first'),
                                                          opp=('opponent_team_location', 'first')).reset_index()
          .sort_values(['team_id', 'date']))
    pl = b[b.played][['season', 'team_id', 'game_id', 'athlete_id', 'athlete_display_name', 'minutes', 'points']]
    out = []
    for (season, team), games in tg.groupby(['season', 'team_id']):
        gids = list(games.game_id); dates = list(games.date)
        tp = pl[(pl.season == season) & (pl.team_id == team)]
        by_game = {g: d for g, d in tp.groupby('game_id')}
        for k in range(5, len(gids)):
            prior = tp[tp.game_id.isin(gids[:k])]
            agg = prior.groupby(['athlete_id', 'athlete_display_name']).agg(gp=('game_id', 'nunique'), mpg=('minutes', 'mean'),
                                                                          ppg=('points', 'mean')).reset_index()
            agg = agg[(agg.gp >= 5) & (agg.gp >= 0.5 * k)]
            if agg.empty:
                continue
            top2 = agg.sort_values('mpg', ascending=False).head(2)
            stars = top2[top2.ppg >= 12]
            here = set(by_game[gids[k]].athlete_id) if gids[k] in by_game else set()
            recent = set(tp[tp.game_id.isin(gids[max(0, k - 5):k])].athlete_id)
            last_game = set(tp[tp.game_id == gids[k - 1]].athlete_id)
            for s in stars.itertuples():
                if s.athlete_id not in here and s.athlete_id in recent:
                    out.append(dict(sport='NCAAM', season=season, game_id=gids[k], date=dates[k], team=games.team.iloc[k],
                                    opp=games.opp.iloc[k], player=s.athlete_display_name, tier=f'{s.ppg:.1f} ppg, {s.mpg:.1f} mpg',
                                    first_game_out=s.athlete_id in last_game))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ NCAAF
def ncaaf_events():
    sch = pd.concat([pd.read_parquet(CB + f'cfb_schedules_cfb_schedules_{y}.parquet') for y in (2025, 2026)])
    sch['date'] = pd.to_datetime(sch.start_date, utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    b = pd.concat([pd.read_parquet(CB + f'cfb_player_box_{y}.parquet') for y in (2025, 2026)])
    b['att'] = b['completions/passingAttempts'].astype(str).str.split('/').str[1].pipe(pd.to_numeric, errors='coerce')
    b['scrim'] = b.rushingYards.fillna(0).astype(float) + b.receivingYards.fillna(0).astype(float)
    p = b.groupby(['season', 'game_id', 'team_id', 'athlete_id', 'athlete_name']).agg(att=('att', 'sum'), scrim=('scrim', 'sum')).reset_index()
    p = p.merge(sch[['game_id', 'date', 'home_id', 'home_team', 'away_id', 'away_team']], on='game_id', how='inner')
    p['team'] = np.where(p.team_id.astype(str) == p.home_id.astype(str), p.home_team, p.away_team)
    p['opp'] = np.where(p.team_id.astype(str) == p.home_id.astype(str), p.away_team, p.home_team)
    tg = p.groupby(['season', 'team_id', 'game_id']).agg(date=('date', 'first'), team=('team', 'first'), opp=('opp', 'first')).reset_index().sort_values(['team_id', 'date'])
    out = []
    for (season, team), games in tg.groupby(['season', 'team_id']):
        gids = list(games.game_id)
        tp = p[(p.season == season) & (p.team_id == team)]
        for k in range(3, len(gids)):
            prior = tp[tp.game_id.isin(gids[:k])]
            agg = prior.groupby(['athlete_id', 'athlete_name']).agg(gp=('game_id', 'nunique'), att=('att', 'sum'), scrim=('scrim', 'sum')).reset_index()
            stars = []
            tot_att = prior.att.sum()
            qb = agg.sort_values('att', ascending=False).head(1)
            if len(qb) and tot_att > 0 and qb.att.iloc[0] / tot_att >= 0.6 and qb.gp.iloc[0] >= 3:
                stars.append((qb.athlete_id.iloc[0], qb.athlete_name.iloc[0], 'QB'))
            sk = agg[(agg.gp >= 3) & (agg.att < 2 * agg.gp)].copy()
            if len(sk):
                sk['ypg'] = sk.scrim / sk.gp
                t = sk.sort_values('ypg', ascending=False).iloc[0]
                if t.ypg >= 70:
                    stars.append((t.athlete_id, t.athlete_name, 'Skill'))
            here = set(tp[tp.game_id == gids[k]].athlete_id)
            recent = set(tp[tp.game_id.isin(gids[max(0, k - 3):k])].athlete_id)
            last_game = set(tp[tp.game_id == gids[k - 1]].athlete_id)
            for aid, name, tier in stars:
                if aid not in here and aid in recent:
                    out.append(dict(sport='NCAAF', season=season, game_id=gids[k], date=games.date.iloc[k], team=games.team.iloc[k],
                                    opp=games.opp.iloc[k], player=name, tier=tier, first_game_out=aid in last_game))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ match to Kalshi
def norm(s):
    s = str(s).lower()
    s = re.sub(r'\(([a-z]+)\)', r' \1', s)
    s = s.replace('&', ' and ').replace("'", '')
    s = re.sub(r'\bst\.?\b', 'state', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    for a, b2 in [('saint', 'state'), ('univ', ''), ('university', '')]:
        pass
    return re.sub(r'\s+', ' ', s).strip()


def match(stars, sport):
    G = pd.read_pickle('games.pkl')
    E = pd.read_pickle('events.pkl')
    g = G[G.sport == sport].copy()
    g['date'] = g.K.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    g['nt'], g['no'] = g.team.map(norm), g.opp.map(norm)
    by_date = {d: x for d, x in g.groupby('date')}
    rows = []
    for s in stars.itertuples():
        cands = pd.concat([by_date.get(d) for d in [s.date, (pd.Timestamp(s.date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                                                       (pd.Timestamp(s.date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')] if d in by_date] or [pd.DataFrame()])
        if cands.empty:
            continue
        nt, no = norm(s.team), norm(s.opp)
        best, bscore, same = None, 0, True
        for c in cands.itertuples():
            a = (fuzz.token_set_ratio(nt, c.nt) + fuzz.token_set_ratio(no, c.no)) / 2
            b2 = (fuzz.token_set_ratio(nt, c.no) + fuzz.token_set_ratio(no, c.nt)) / 2
            sc, sm = (a, True) if a >= b2 else (b2, False)
            if sc > bscore:
                best, bscore, same = c, sc, sm
        if best is None or bscore < 80:
            continue
        flip = (lambda x: x) if same else (lambda x: 1 - x)
        m, cv = best.m, best.cv
        ev = E[E.ticker == best.ticker]
        p_ref = ref_h = vol_ref = np.nan
        if len(ev):
            e = ev.iloc[0]
            ref_h, vol_ref = e.ref_h, e.vol_ref
            p_ref = m[H0 - int(ref_h)]
        rows.append(dict(**s._asdict(), ticker=best.ticker, kalshi_team=best.team if same else best.opp, match_score=bscore,
                         ref_h=ref_h, vol_ref=vol_ref, p_ref=flip(p_ref), p_24=flip(best.p_24), p_close=flip(best.p_close),
                         won=best.won if same else 1 - best.won, pre_vol=best.pre_vol))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    mb, fb = ncaam_events(), ncaaf_events()
    print('star-out team-games found: NCAAM', len(mb), ' NCAAF', len(fb))
    M = pd.concat([match(mb, 'NCAAM'), match(fb, 'NCAAF')], ignore_index=True).drop(columns=['Index'], errors='ignore')
    # one row per team-game (two stars out on one team -> keep both names)
    M = (M.groupby(['ticker', 'kalshi_team'], as_index=False)
          .agg(sport=('sport', 'first'), season=('season', 'first'), date=('date', 'first'), team=('team', 'first'), opp=('opp', 'first'),
               players=('player', lambda x: '; '.join(x)), tier=('tier', lambda x: '; '.join(x)), first_game_out=('first_game_out', 'max'),
               ref_h=('ref_h', 'first'), vol_ref=('vol_ref', 'first'), p_ref=('p_ref', 'first'), p_24=('p_24', 'first'),
               p_close=('p_close', 'first'), won=('won', 'first'), pre_vol=('pre_vol', 'first'), match_score=('match_score', 'min')))
    M['both_out'] = M.groupby('ticker').ticker.transform('count') > 1
    M.to_pickle('college_stars.pkl')
    print('matched to Kalshi:', M.groupby('sport').size().to_dict())
    print(M.sample(min(8, len(M)), random_state=1)[['date', 'team', 'kalshi_team', 'opp', 'players', 'match_score']].to_string())
