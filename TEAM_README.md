# More than a Hack 2027 — Ball Possession

This repository contains a reproducible baseline for predicting possession in every frame of the
BAR-ATH tactical video. It uses the official Metrica starter algorithm and the authorised local
tracking bundle. Local match data and our submission candidate are excluded from Git. The four
organizer-provided baseline reference CSVs remain tracked as in the official repository.

## Generate the official variants

Requirements: Python 3.8 or newer, NumPy, and pandas. The environment used for this preparation
was Python 3.14.10, NumPy 2.5.4, and pandas 2.3.3. The optional notebook dependencies are not
needed for generation.

From PowerShell at the repository root, with the authorised data files in `data/tactical/`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter\baseline.py
Get-ChildItem evaluation\results\baseline_*.csv | ForEach-Object {
    .\.venv\Scripts\python.exe evaluation\validate_submission.py $_.FullName
}
New-Item -ItemType Directory -Force submission_artifacts | Out-Null
Copy-Item evaluation\results\baseline_smooth50.csv submission_artifacts\TEAM_NAME_PENDING.csv -Force
.\.venv\Scripts\python.exe evaluation\validate_submission.py submission_artifacts\TEAM_NAME_PENDING.csv
```

The baseline writes `baseline_raw`, `baseline_smooth25`, `baseline_smooth50`, and
`baseline_smooth125` into `evaluation/results/`. Each output has one row per tactical video frame.
The submitted candidate is the 50-frame half-window variant. The public five-minute labels were
not used for training or tuning.

## Experimental V2 model

`team_model.py` contains a separate tracking-evidence and temporal-state model. It has not been
shown to outperform the official baseline. See [MODEL_V2_NOTES.md](MODEL_V2_NOTES.md) for its
features, fixed defaults, tests, diagnostics, and reproduction command. Reproduce the hardened
candidate with:

```powershell
.\.venv\Scripts\python.exe team_model.py --out submission_artifacts/TEAM_NAME_PENDING_v2_hardened.csv
.\.venv\Scripts\python.exe evaluation\validate_submission.py submission_artifacts/TEAM_NAME_PENDING_v2_hardened.csv
```

Its full-match candidate is gitignored. Optional Smart Tagging and homography files, when present,
belong in the ignored `data/tactical/` folder; the current V2 implementation does not consume them.

## Submission reminder

Before uploading, replace the provisional filename with the exact registered team name, publish
the reviewed code repository yourself, and submit the CSV and technical report through the
organisers' Google Form. See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) and
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
