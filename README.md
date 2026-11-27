# Sleeper NBA decision engine

This engine reads your Sleeper lock-in league and emails you three things: tonight's lineup, a lock-or-ride call for every open score, and add/drop and trade ideas. It chooses every option by its effect on **P(win this week)** and on playoff odds, not on raw points.

```
python main.py --mode pregame --dry-run      # build today's email, print it, send nothing
python main.py --mode auto                   # what the scheduler runs every 30 min
python main.py --backtest --weeks 1-5        # replay your past lock decisions
python -m pytest -q
```

## Layout
| path | what |
|---|---|
| `data/sleeper.py` | Read-only API client with retries, timeouts and a file cache (`data_raw/`, gitignored). Falls back to the cache and records the failure. |
| `data/loader.py` | One snapshot of the league: rosters, matchups, per-game logs, projections, schedule, and tip times (Central) from the ESPN scoreboard. |
| `model/scoring.py` | Scores a stat line with the league's `scoring_settings`, read at runtime. Bonuses are included. |
| `model/distributions.py` | Per-player, per-game FP distribution (see the math below). |
| `optimize/lock.py` | Lock rules, lock inference, and backward induction. |
| `optimize/lineup.py` | Joint slot/lock planner: day-by-day Hungarian assignment. |
| `optimize/montecarlo.py` | Monte Carlo of the rest of the week with common random numbers, plus the lock/ride decisions. |
| `optimize/season.py` | Season sim, giving playoff odds. |
| `optimize/waivers.py`, `optimize/trades.py` | Add/drop and trade search. |
| `optimize/backtest.py` | Replays past weeks. |
| `report/` | Email (HTML plus plain text) and Gmail SMTP. |
| `state/sent.json` | What has already been sent, so alerts don't repeat. Not in git: persisted between runs with the Actions cache. |
| `state/locks.json` | Locks you reported (email, workflow input, or hand edits). Not in git, same as above. |
| `data/sleeper_private.py` | Optional read-only Sleeper GraphQL lock-state reader (`SLEEPER_TOKEN`). |
| `data/locks.py` | Lock-reply parsing, fuzzy name matching, the Gmail IMAP reader. |
| `config.yaml` | Every tunable. |

## Setup (GitHub Actions + Gmail)
1. Create a GitHub repo (public or private) and push this folder. Nothing personal is committed: the email address, tokens and `state/` live only in secrets or the Actions cache.
2. Turn on 2-Step Verification for the Gmail account at myaccount.google.com → Security.
3. Create an App Password at myaccount.google.com → Security → App passwords ("Mail"). You get a 16-character code. Gmail only allows app passwords once 2-Step Verification is on.
4. In the repo, go to Settings → Secrets and variables → Actions and add the secrets below, plus `SLEEPER_TOKEN` (optional; see "Reading lock state from Sleeper"):
   - `GMAIL_USER`: the sending Gmail address.
   - `GMAIL_APP_PASSWORD`: the 16-character code.
   - `EMAIL_TO` (optional): where alerts go. If unset, alerts go to your Gmail address (`GMAIL_USER`).
5. Nothing else to configure. The workflow only needs read access (`contents: read`). `state/` is saved and restored with `actions/cache`, never committed, and Actions caches can't be downloaded by the public.
6. Go to Actions → sleeper-alerts → Run workflow, set mode `pregame`, and run it once to test.

The cron runs every 30 min in UTC. `main.py --mode auto` does the timing itself in America/Chicago, so daylight-saving changes are handled:
- **Pregame:** sent on the first run that falls 0–75 min before the day's first tip, at most once per day.
- **Postgame:** sent when one of your players' games has gone final and its stat lines are posted. If another of your games is due to finish within about 25 min, it waits so both are covered in one email. Nothing is sent if there's no open call.

Each run costs about 1 Actions minute. Every 30 minutes that adds up to roughly 1,450 min/month, which is under the 2,000 free minutes for a private repo. If you need to save minutes, set the cron to `*/30 13-23,0-6 * * *`.

Credentials only ever come from environment variables. The repo contains none.

## Reading lock state from Sleeper (optional, recommended)
Sleeper's public API has no lock field. The website does have one, through its private GraphQL API:
`POST https://sleeper.com/graphql`, query `get_matchup_legs`, which returns `matchup_legs(league_id, round) { roster_id starters starters_games … }`.
- `starters_games` maps player_id → game_id for every **locked** starter. Unlocked starters are absent.
- I verified it on week 6, 2026-11-27: every entry's game is exactly the game whose score the public API shows for that player.
- When the token is available, the engine treats this map as authoritative. It overrides inference and email reports, and any disagreement is flagged.
- The engine only ever sends this one read-only query. It never sends a mutation.

**Getting the token** (it's tied to your account):
1. Log in at sleeper.com in a desktop browser.
2. Open DevTools (F12), go to the Network tab, and filter by `graphql`.
3. Click around your league, pick any `graphql` request, and copy the value of its **`authorization`** request header. It's a long string starting with `eyJ`.

**Where to put it:**
- Locally: in `.env` in this folder, as `SLEEPER_TOKEN=eyJ…`. `.env` is gitignored.
- GitHub Actions: as the repo secret `SLEEPER_TOKEN`.
- The code reads it only from there. It is never printed, logged or committed, and is sent only to sleeper.com.

**Refreshing it:** the token is a JWT with an expiry date, typically about a year after it was issued.
- The email warns under Assumptions when it is less than 14 days from expiring.
- If the query fails (bad or expired token, HTTP error, schema change), the engine falls back to inference and email reports and says so under Assumptions.
- To refresh, repeat the three steps above and replace the value. Logging out of Sleeper everywhere may also invalidate it.
- To turn the feature off, set `private_api.enabled: false` in config.

## Telling the engine what you've locked (fallback)
The API has no lock field. Locks are inferred where the evidence allows, and you can also report them directly. Any of these three ways works:

1. **Reply to an alert email** with `locked giannis, jerome`. `lock AD and bam` and `Locked: Maxey` also work.
   - Before each run, the engine reads Gmail over IMAP (`imap.gmail.com:993`, read-only) using the same `GMAIL_USER` / `GMAIL_APP_PASSWORD`. It looks for messages **from** your alert address (`EMAIL_TO`, else your Gmail address) whose subject starts with `Re: Sleeper |` or contains `locked`, received since Monday 00:00 CT.
   - Only the new text above the quoted reply is read. "unlocked …" is ignored, because locks are irreversible.
   - Each Message-ID is processed once.
   - Gmail must have IMAP enabled: Settings → Forwarding and POP/IMAP. It is on by default for new accounts.
2. **Workflow input:** Actions → sleeper-alerts → Run workflow, then fill `locked` with something like `giannis, jerome`. Locally the same thing is `python main.py --locked "giannis, jerome"`.
3. **Edit `state/locks.json` by hand:**
   ```json
   {"weeks": {"6": {"1380": {"game_id": "1393301049854418944", "date": "2026-11-25", "score": 65.0,
                             "source": "file", "msg_id": null, "reported_at": "2026-11-25T22:10:00-06:00"}}},
    "processed": []}
   ```
   The keys are the week number and then the Sleeper player_id. Only `score` is required; `date` is shown in the email.

**Name matching.** Names are matched against **your roster only**: full name, last name, first name, initials (`AD`), the `locks.nicknames` list in config, and a difflib fuzzy match with cutoff 0.8.
- If a name is ambiguous (for example `johnson` when you have Keldon and Cameron), nothing is recorded and the email says so.

**Which game gets locked.** A reported lock applies to the player's most recent game that had finished when you sent the report, taking a game as over 2 h after tip.

**How the engine uses it.**
- A reported lock fixes that player at the locked score, freezes him in his slot, and removes him from the lock/ride calls.
- Anything you haven't reported is still inferred.
- If the API contradicts a report (it shows a different locked score, or it is counting a later game), the API wins and the email flags the conflict under Assumptions.
- The lineup section lists `Locked (reported/inferred)`, and every email ends with "Reply 'locked name, name' to record locks."

## Public-repo safety
- **No personal data in git:** no email address, tokens or `state/`. The recipient comes from secrets.
- **Quiet CI logs:** in CI (`CI=true`, or `--quiet`) the program prints only the mode, sent / nothing to send / failed (error type and source, with emails scrubbed), and timings. It prints no players, scores, lineups, locks, team names or P(win), because Actions logs on a public repo are public.

## Lock-in rules implemented
- Each starter counts exactly one game per week. Once a game ends, it can be locked any time before that player's next tip. A lock is irreversible and freezes the slot.
- An unlocked starter at week end gets his team's final game, even a DNP (= 0).
- A game counts, and can be locked, only if he was in a starting slot while it was played. Bench games never count.
- Unlocked players can move between eligible slots: G = PG/SG, F = SF/PF, UTIL = anyone.
- The API has no lock field, so locks are inferred. If a starter's `starters_points` equals an earlier game's FP after a later game was played, that earlier game was locked. A counting score whose next game hasn't tipped is reported as an open call.

## The math (finance analogies)
**Per-game distribution.**
- The level is a weighted blend of three estimates: an exponentially weighted recency mean (half-life 7 games over the last 15), the season mean, and Sleeper's projection (weights 0.45 / 0.25 / 0.30).
- The shape is the player's own empirical outcome ratios (game ÷ season mean), rescaled to that level. This is a bootstrap, so it keeps fat right tails.
- Players with fewer than 8 games are credibility-weighted toward a positional prior.
- A small opponent adjustment (FP allowed vs. position, shrunk, capped at ±5%) is applied.
- P(play) comes from the injury tag (configurable). A DNP scores 0.

**Lock decision = American option / optimal stopping.** For one slot facing games X₁…Xₙ:
`Vₙ = E[Xₙ]`, `Vₖ = E[max(Xₖ, Vₖ₊₁)]`. You lock game k if Xₖ ≥ Vₖ₊₁, the continuation value. With several players and slots, the planner works backwards day by day. Each day it solves an assignment problem: which playing player goes in which open slot, maximising the increase in slot values. A slot can be "streamed" (A on Friday, B on Sunday), and each player stays in one slot for the week.

**Objective = P(win), not points.**
- 10,000 seeded simulations of the rest of the week, for you and for your opponent. The opponent is modelled with the same planner: best lineup plus a threshold policy.
- Every open call is decided by comparing P(win | lock) with P(win | ride), using common random numbers.
- So variance preference falls out on its own. When you're favoured you lock more, because you're short volatility. When you're the underdog you ride more, because you're long volatility.

**Season / playoff odds.**
- A team's weekly score is Normal(μ, σ). μ is the planner's expected score for that team's actual NBA schedule that week, so game counts matter. σ is the league's empirical week-to-week SD.
- The current week uses the full Monte Carlo totals.
- Standings are by record, then points-for. The top 6 make the playoffs, and both division winners are assumed to qualify.

**Waivers.**
- Each top free agent is tested against your lowest-value droppable player.
- ΔP(win) counts only the games he can actually start: after he becomes available, before tip.
- Δ playoff odds comes from the season sim.
- A move is recommended only if it adds ≥ 1pp playoff odds or ≥ 3pp P(win).
- The FAAB bid applies only when the player is on waivers: `bid = clamp(k · Δplayoff_pp · remaining_budget / weeks_left, min_bid, max_frac · remaining)`. This spends in proportion to the playoff percentage points the player buys, relative to how many weeks the budget has to last.

**Trades.**
- Every 1-for-1 and 2-for-1 (both directions) with every team is screened with a cheap proxy: the best legal lineup of each player's average weekly lock-in value, plus 0.25 × bench depth.
- The top 20 get a 3,000-run season sim, and the top 8 get the full 10,000.
- A trade is flagged (★) if your playoff odds rise ≥ 2pp. Trades that cost the other side > 3pp are dropped as unrealistic.

## Tuning
Everything lives in `config.yaml`: injury probabilities, blend weights, half-life, sim counts, thresholds and FAAB parameters. Raise `sims.n_week` for steadier P(win); runtime grows roughly linearly with it.

## Known limitations
- Lock state is inferred unless you report it. If you locked a score that is still your latest game and didn't report it, the email will still show it as an open call.
- The backtest assumes each final starter sat in his slot all week. The API keeps only the final lineup, and historical injury tags aren't available.
- Sleeper's matchup endpoint has served different versions of the same week on back-to-back requests. The client takes the most common of 3 cache-busted fetches and flags any disagreement in Assumptions.
- Waiver timing (`waiver_after_game_start`, 2 clear days) is approximated. A free agent counts as an instant add unless he was dropped in the last 2 days or his game today has already tipped.
