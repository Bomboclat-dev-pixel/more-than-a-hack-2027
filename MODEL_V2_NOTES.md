# Possession Model V2 Notes

## Status and data use

V2 is a separate, explainable tracking model in `team_model.py`. It does not modify
`starter/baseline.py` or the V1 candidate. It reads only the locally available tactical ATD text
and metadata XML. The Smart Tagging XML and homography CSV were found in the parent `TACTICAL`
directory and copied unchanged into ignored `data/tactical/`. This V2 run uses ATD only. Smart
Tagging contains automatic tactical intervals (including 59 set-piece intervals) and could offer
weak restart context, but the intervals do not state framewise possession or referee ball-in-play
status. The model therefore does not convert them into D or team labels. The homography maps pitch
coordinates to image coordinates; normalized ATD coordinates already support metric pitch distances,
so homography adds no needed feature here. Neither optional file is parsed by the model. It does not
read either video, broadcast tracking, public ground truth, or private
ground truth. No match events, manual observations, annotations, or corrected tracking were used.

No accuracy, macro-F1, or spellF1 result is claimed. The public evaluator was not run. The official
submission validator was run only to check the output contract.

## Architecture

The model processes each tactical ATD frame, builds home/away evidence, then decodes it through a
state machine with persistence. It emits `H`, `A`, `D`, or `X`. It is not a nearest-player label
followed by a centred majority filter.

### Team mapping and distances

The XML's first team is home (Barcelona) and second is away (Athletic Club). Each track slot is
mapped by its XML `teamId`; slots are treated as short-lived tracklets, not persistent player
identities. For ball position `(bx, by)` and player position `(px, py)`, distance is computed in
metres as:

```text
sqrt(((px - bx) * 105.3)^2 + ((py - by) * 68.0)^2)
```

Each frame records the nearest distance for each team, not just the globally nearest player. A
one-sided tracking frame is weak evidence: its total team score is capped below the classification
threshold.

### Local support and movement features

The team evidence score combines:

- Nearest-distance margin: `tanh((away_nearest - home_nearest) / 4 m)`.
- Local support: per-team sum of `exp(-distance / 4 m)` across observed players.
- Local numbers: per-team counts within 4.5 m of the ball.
- Approach: change in the same slot's player-to-ball distance between adjacent frames, converted
  to m/s and clipped at 12 m/s. A tracklet is used only when it exists in both adjacent frames and
  its derived movement is physically plausible.
- Ball/team movement alignment: cosine alignment of the ball's consecutive-frame displacement
  with nearby players' same-slot displacement, weighted by distance. This is only available when
  both ball frames are present and displacement is plausible.
- Local activity: nearby players' provided ATD speeds, scaled by 5 m/s and weighted by proximity.

The combined signed score is:

```text
1.30 * distance_margin
+ 0.55 * support_margin
+ 0.25 * nearby_count_margin
+ 0.35 * approach_margin
+ 0.25 * movement_alignment_margin
+ 0.15 * local_activity_margin
```

Evidence of at least 0.72 in magnitude is required to propose a team state. These are documented
physical scales and fixed defaults; no label-driven search was performed.

### Temporal state decoder

The decoder resets at the official start of each graded half and emits X on all ungraded frames,
using the published `games.half_video_blocks` alignment. This prevents state from carrying across
halftime or pre/post-match gaps. It requires 18 consecutive frames (0.72 seconds) of strong opposing evidence before
switching H/A. Once confirmed, it assigns the transition back to the start of that uninterrupted
evidence run. Initial team assignment requires five frames; moving from a confirmed D state to a
team requires five frames. Weak or missing evidence preserves the current live state for up to 12
frames, then outputs `X`. The previous state is kept internally so evidence can resume without a
one-frame state reset.

`X` is used before a confident state and after prolonged uncertainty. Since the official spellF1
path forward-fills X, long X stretches can carry the prior label across a true transition; this is
a known risk and no accuracy claim is made for the selected persistence defaults.

### Dead ball and missing observations

Missing ball observations do **not** by themselves produce D. After a confirmed D state, missing
ball may retain D for up to 30 frames, then becomes X. Positive dead-ball candidates are:

1. Ball coordinates more than 1% outside the normalized pitch bounds, persistent for three frames.
2. Ball displacement at most 0.04 m/frame and speed at most 0.20 m/s for at least one second,
   while the nearest tracked player is farther than 4 m; the unattended condition is confirmed
   for three more frames before decoding D.

These tracking-only rules are proxies, not referee state. A stationary live ball or a tracking
error can still be misclassified. The absent Smart Tagging file means set-piece context and
team/type tags were not used.

## Output and label-free diagnostics

Hardened candidate: `submission_artifacts/TEAM_NAME_PENDING_v2_hardened.csv`. It was regenerated
after correcting parsing of NaN ball coordinates (they now count as missing) and adding per-half
state resets. The earlier V2 candidate remains untouched.

Final run on the local full tactical ATD:

| Diagnostic | Result |
|---|---:|
| Frames | 199,319 |
| H / A / D / X | 45,399 / 32,820 / 4,273 / 116,827 |
| Raw contiguous H/A spells (X breaks a spell) | 1,002 |
| Mean / median / p90 raw spell duration | 3.12 s / 2.04 s / 6.80 s |
| Ball observed / missing | 113,488 (56.94%) / 85,831 |
| Missing-ball frames labelled H / A / D / X | 435 / 695 / 452 / 84,249 |
| X on visible / missing-ball frames | 32,578 / 84,249 |
| D runs (count; mean / median / p90) | 81; 2.11 s / 1.52 s / 4.76 s |
| Stationary-unattended evidence frames | 1,013 |
| Outside-pitch evidence frames | 3,897 |
| State transitions between adjacent non-X frames | 286 (92 H→A; 92 A→H) |
| Feature extraction / decoding / total runtime | 31.79 s / 0.17 s / 32.24 s |

Per-half label counts are below; ungraded frames are all X by design:

| Region | H | A | D | X | H/A spells; mean / median / p90 |
|---|---:|---:|---:|---:|---|
| First half | 23,648 | 15,971 | 1,663 | 38,185 | 509; 3.11 / 2.00 / 6.92 s |
| Second half | 21,751 | 16,849 | 2,610 | 38,180 | 493; 3.13 / 2.12 / 6.76 s |
| Ungraded | 0 | 0 | 0 | 40,462 | — |

D/X contiguous run durations (count; mean / median / p90 seconds) are separated by region so
ungraded X stretches do not distort the in-play summary:

| Region | D runs | X runs |
|---|---|---|
| First half | 28; 2.38 / 1.50 / 5.07 | 400; 3.82 / 0.52 / 9.51 |
| Second half | 53; 1.97 / 1.52 / 4.66 | 398; 3.84 / 0.64 / 10.01 |
| Ungraded | 0 | 3; 539.49 / 447.40 / 734.47 |

The shipped ATD-only baseline outputs, measured without labels, show these total H/A spell counts
and durations: raw 2,265 (mean 2.00 s, median 0.96 s, p90 5.04 s), smooth25 851 (5.68 / 3.12 /
13.64 s), smooth50 669 (7.64 / 4.72 / 19.24 s), smooth125 430 (13.48 / 8.56 / 34.24 s).
V2's 1,002 short spells and 58.6% X share show substantial fragmentation/abstention compared with
smooth50. These are label-free segmentation comparisons only; they do not measure accuracy.

Transitions are assigned only after sustained evidence: an H/A change needs 18 consecutive frames
of strong opposing evidence, with the transition retrospectively placed at the run's first frame.
The reported non-X transitions are therefore not one-frame flips. The short spells mostly reflect
X gaps and fragmented short runs; they cannot be judged correct without labels. These are
operational diagnostics only. Spell durations count contiguous raw H/A output; X breaks
these diagnostic runs. They are not evaluator scores or evidence of model quality. A notable
limitation is the large abstention share and substantial raw spell fragmentation; evaluating
either against labels was intentionally not done.

## Tests and validator

Eight standard-library `unittest` cases passed using synthetic inputs only. They cover missing-ball
handling including NaN coordinates, short versus persistent H/A evidence, affirmative dead-ball
evidence, missing ATD observations, half/gap state reset, and output CSV formatting.

The official validator passed the V2 CSV: 199,319 rows, frames 0 through 199318, required columns
and allowed labels. This verifies format only.

## Reproduction

From the repository root in PowerShell, with the authorized ATD files under `data/tactical/`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe team_model.py --out submission_artifacts/TEAM_NAME_PENDING_v2_hardened.csv
.\.venv\Scripts\python.exe evaluation\validate_submission.py submission_artifacts/TEAM_NAME_PENDING_v2_hardened.csv
```

The official generator is not needed for V2. The script uses the data paths in `games.py` and
respects `MTAH_DATA_DIR` through the standard repository data helper. The submission artifact
directory's `*.csv` files are gitignored.
