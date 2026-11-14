# Technical Report: BAR-ATH Ball Possession Baseline

## 1. Problem

Predict one label for every one of the 199,319 tactical video frames: `H` for Barcelona in
possession while play is live, `A` for Athletic Club in possession while live, `D` for a dead ball,
or `X` for abstention. The ranking metric is possession spell F1; macro F1 is also reported.

## 2. Dataset and inputs

The implementation reads the authorised local files
`data/tactical/BAR-ATH_tactical_atd.txt` and
`data/tactical/BAR-ATH_tactical_atd.xml`. It does not read the video, broadcast feed, homography,
Smart Tagging, or any ground truth. The raw ATD contains one tactical record for each video frame.
Records are numbered 1 through 199,319; the starter maps record `frame_id` to submission frame
`frame_id - 1`. The file uses normalised 0–1 coordinates and a player-track list plus a ball group.
Metadata describes a 105 by 68 metre pitch. The starter's match constants use 105.3 by 68.0 m.

The XML lists Barcelona first and Athletic Club second and assigns each track slot to a team.
There are 46 slots for 22 players; slots are temporary tracklets rather than stable player
identities. The original source files were left untouched and are ignored by Git.

No model is trained, and no labelled training data is used. No human labels, manual observations,
or frame-specific annotations from this match were used. The public five-minute ground truth was
not used for training, tuning, or candidate selection.

## 3. Algorithm architecture

The selected candidate is the official, unmodified `starter/baseline.py`, with its published
50-frame smoothing half-window. For each frame with a visible ball, it finds the closest observed
player in pitch-scaled coordinates and assigns that player's team. The raw H/A signal is smoothed
with a centred majority vote over up to 101 frames (the current frame plus up to 50 on either
side). It writes `H` or `A` according to the larger vote; ties go to `H`.

## 4. Player-to-ball distance

For each tracked player, the starter computes squared Euclidean distance in metres:
`((player_x - ball_x) * pitch_length)^2 + ((player_y - ball_y) * pitch_width)^2`.
The closest observed player wins. Squared distances preserve the ordering of Euclidean distances
and avoid a square root. Coordinates with missing observations are skipped.

## 5. Team mapping

The starter reads team IDs and player slot assignments from the XML. The first listed team is
home, mapped to `H` (Barcelona); the second is away, mapped to `A` (Athletic Club). It does not
infer team from direction of attack or slot number.

## 6. Missing ball handling

When the ball coordinate is missing, or no player slot is observed, the raw code is “nobody.” For
each frame the smoother counts only H and A observations in its window. If neither appears, it
emits `D`; otherwise it emits the team with more votes. This is the baseline's proxy for a dead
ball. It does not distinguish a real stoppage from a live ball that the tracker failed to observe.

## 7. Temporal smoothing

The 50-frame half-window is a centred majority filter, approximately two seconds of context on
each side at 25 fps. Cumulative sums make the vote efficient. The baseline variants use half-window
sizes 0, 25, 50, and 125. The selected size is the organisers' published full-match baseline
configuration; it was not selected by tuning against the public five-minute ground truth.

## 8. H/A/D/X logic

`H` and `A` come from nearest-player team votes. `D` is emitted where a smoothing window has no
visible-ball-derived team votes. The baseline never emits `X`; it always supplies a label.

## 9. Limitations and assumptions

Nearest-player distance is only a proxy for control. A nearby opponent may be challenging, the
ball may be in flight or loose, and tracking errors or identity switches may change the nearest
slot. A centred filter uses future as well as past frames and can blur transition timing. Missing
ball is treated indirectly through the vote and is not reliable dead-ball detection. No player
identity continuity, velocity, Smart Tagging, or explicit possession state model is used.

Possible future work would add a causal or semi-Markov possession model, explicit ball-visibility
and movement evidence, player motion continuity, and Smart Tagging as a noisy prior. None of these
is implemented or claimed as an improvement here. Any candidate should be assessed without
tuning against the public ground truth and compared using legitimate labelled data.

## 10. Results

The challenge data documentation reports ball observations on 113,488 of 199,319 total frames
(56.9%), and on 94,214 of the 158,857 graded frames (59.3%). Thus 40.7% of graded frames have no
ball observation, rounded to 41% in the challenge README. The README's 41% uses the graded-frame
denominator; it is consistent with 56.9% observed over the full video.

The starter parser's printed 113,488 count is computed as `codes != NOBODY`, so it counts frames
where the parser assigned a nearest tracked player's team. The parser leaves `NOBODY` when the ball
coordinate is missing and also when no player slot is observed. A separate read-only count of the
raw ball field found 113,488 observations overall and 94,214 in the graded intervals; these match
the challenge data documentation for this file. The parser's count is therefore reported as an
assignment count, not treated as a general ball-visibility test.

All four generated CSV variants passed the official format validator: raw, smooth25, smooth50,
and smooth125. The selected `smooth50` file has 199,319 rows and reports 669 H/A spells with a mean
duration of 7.6 seconds according to the validator's format-only summary. That spell count is
descriptive, not a measure of correctness.

The organisers report full-match spellF1 0.2433 for their published smooth50 baseline. This is a
reference result, not a locally measured result. No full-match labels are available locally, and
the public five-minute ground truth was not used to tune or claim a score. The official validator
checks the CSV contract only, not prediction quality.

## 11. Reproduction

From the repository root in PowerShell, with the authorised tactical ATD files in `data/tactical/`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter\baseline.py
.\.venv\Scripts\python.exe evaluation\validate_submission.py evaluation\results\baseline_smooth50.csv
New-Item -ItemType Directory -Force submission_artifacts | Out-Null
Copy-Item evaluation\results\baseline_smooth50.csv submission_artifacts\TEAM_NAME_PENDING.csv -Force
.\.venv\Scripts\python.exe evaluation\validate_submission.py submission_artifacts\TEAM_NAME_PENDING.csv
```

The generator and validator scripts were not modified. `requirements.txt` lists NumPy and
constrains pandas to the 2.x series (`pandas<3`), which was used to verify the official validator
on this Windows environment. Local match data and the team's submission candidate are gitignored.
The four organizer-provided baseline reference CSVs remain tracked by the official repository
configuration.
