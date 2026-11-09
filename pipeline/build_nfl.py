"""Build the NFL star-injury event table (2023 season to date) from nflverse files.

Star definitions (fixed before looking at results):
  QB    : team's primary passer in prior games this season (>=60% of team pass attempts, >=3 prior games)
  Skill : WR/TE/RB who leads his team in scrimmage yards per game in prior games
          AND averages >=70 scrimmage yards per game (>=3 prior games)
Event   : star listed Out or Doubtful on the final weekly report AND has no stat line that week.
Line    : nflverse closing spread and moneyline (the line after the news is known).
"""
import pandas as pd, numpy as np

RAW = "raw/"
YEARS = [2023, 2024, 2025, 2026]

games = pd.read_csv(RAW + "games.csv")
games = games[(games.season >= 2023) & games.result.notna() & games.spread_line.notna()].copy()

stats = pd.concat([pd.read_csv(RAW + f"sp_{y}.csv", low_memory=False) for y in YEARS])
stats["scrim"] = stats[["rushing_yards", "receiving_yards"]].fillna(0).sum(axis=1)
stats["season_type"] = stats["season_type"].fillna("REG")

inj = pd.concat([pd.read_csv(RAW + f"injuries_{y}.csv") for y in YEARS])
inj = inj[inj.report_status.isin(["Out", "Doubtful"])]

# team pass attempts per game
team_att = stats.groupby(["season", "week", "team"]).attempts.sum().rename("team_att").reset_index()
stats = stats.merge(team_att, on=["season", "week", "team"], how="left")

def star_flags(season, week, team):
    """Return dict player_id -> (tier, name) of stars for team entering this week."""
    prior = stats[(stats.season == season) & (stats.week < week) & (stats.team == team)]
    if prior.week.nunique() < 3:
        return {}
    out = {}
    qb = prior[prior.position == "QB"].groupby(["player_id", "player_display_name"]).agg(
        att=("attempts", "sum"), g=("week", "nunique")).reset_index()
    tot_att = prior.drop_duplicates("week").team_att.sum()
    if len(qb) and tot_att > 0:
        top = qb.sort_values("att").iloc[-1]
        if top.att / tot_att >= 0.6 and top.g >= 3:
            out[top.player_id] = ("QB", top.player_display_name)
    sk = prior[prior.position.isin(["WR", "TE", "RB"])].groupby(
        ["player_id", "player_display_name", "position"]).agg(
        yds=("scrim", "sum"), g=("week", "nunique")).reset_index()
    sk = sk[sk.g >= 3]
    if len(sk):
        sk["ypg"] = sk.yds / sk.g
        top = sk.sort_values("ypg").iloc[-1]
        if top.ypg >= 70:
            out[top.player_id] = (top.position, top.player_display_name)
    return out

played = set(zip(stats.season, stats.week, stats.player_id))
inj_set = inj.groupby(["season", "week", "team"]).gsis_id.apply(set).to_dict()

rows = []
for g in games.itertuples():
    for side, team, opp in [("home", g.home_team, g.away_team), ("away", g.away_team, g.home_team)]:
        stars = star_flags(g.season, g.week, team)
        listed = inj_set.get((g.season, g.week, team), set())
        for pid, (tier, name) in stars.items():
            if pid in listed and (g.season, g.week, pid) not in played:
                rows.append(dict(game_id=g.game_id, season=g.season, week=g.week, game_type=g.game_type,
                                 gameday=g.gameday, team=team, opp=opp, side=side,
                                 player=name, tier=tier))

ev = pd.DataFrame(rows)
# one row per team-game (if two stars out, keep both names, tier = QB if any QB)
ev = ev.groupby(["game_id", "team"]).agg(
    season=("season", "first"), week=("week", "first"), game_type=("game_type", "first"),
    gameday=("gameday", "first"), opp=("opp", "first"), side=("side", "first"),
    players=("player", lambda x: "; ".join(x)),
    tier=("tier", lambda x: "QB" if "QB" in set(x) else "Skill")).reset_index()

ev = ev.merge(games[["game_id", "home_score", "away_score", "result", "spread_line",
                     "home_moneyline", "away_moneyline"]], on="game_id")

# Convert to the injured team's perspective. nflverse: spread_line > 0 means home favored;
# result = home - away.
sign = np.where(ev.side == "home", 1, -1)
ev["team_margin"] = ev.result * sign
ev["team_spread"] = -ev.spread_line * sign            # betting-style: negative = favored
ev["ats_margin"] = ev.team_margin + ev.team_spread    # >0 covers, 0 push
ev["covered"] = np.select([ev.ats_margin > 0, ev.ats_margin < 0], ["Cover", "Loss"], "Push")
ev["won"] = (ev.team_margin > 0).astype(int)

def implied(ml):
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
ph, pa = implied(ev.home_moneyline), implied(ev.away_moneyline)
p_team = np.where(ev.side == "home", ph / (ph + pa), pa / (ph + pa))   # vig removed
ev["implied_win_prob"] = p_team.round(3)

# both teams missing a star in the same game -> flag
cnt = ev.groupby("game_id").team.transform("count")
ev["both_teams_missing_star"] = cnt > 1

# first game missed vs repeat absence (same player string, consecutive appearance)
ev = ev.sort_values(["team", "season", "week"])
ev["first_game_out"] = ~ev.duplicated(["team", "season", "players"])
stats = stats.copy()

ev["score_team_opp"] = np.where(ev.side == "home",
                       ev.home_score.astype(int).astype(str) + "-" + ev.away_score.astype(int).astype(str),
                       ev.away_score.astype(int).astype(str) + "-" + ev.home_score.astype(int).astype(str))
ev.to_csv("nfl_star_injury_events.csv", index=False)

# All-games baseline: underdog/favorite ATS from same seasons
print("events:", len(ev), "| clean (only one side missing a star):", (~ev.both_teams_missing_star).sum())
clean = ev[~ev.both_teams_missing_star]
def summary(df, label):
    dec = df[df.covered != "Push"]
    n = len(dec); c = (dec.covered == "Cover").sum()
    from scipy.stats import binomtest
    p = binomtest(c, n, 0.5).pvalue if n else np.nan
    print(f"{label:28s} n={len(df):4d}  cover%={c/n*100 if n else 0:5.1f}  "
          f"win%={df.won.mean()*100:5.1f}  implied win%={df.implied_win_prob.mean()*100:5.1f}  "
          f"avg ATS margin={df.ats_margin.mean():+5.2f}  p(vs 50%)={p:.3f}")
summary(clean, "All star-out")
summary(clean[clean.tier == "QB"], "QB out")
summary(clean[clean.tier == "Skill"], "Skill star out")
summary(clean[clean.first_game_out], "First game missed")
summary(clean[~clean.first_game_out], "Repeat absence")
for s in sorted(clean.season.unique()):
    summary(clean[clean.season == s], f"Season {s}")
