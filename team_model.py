"""Explainable tracking-based possession model (V2).

This model is separate from starter/baseline.py. It combines teamwise proximity and local
support, short-term approach and movement alignment, a persistent possession decoder, and
conservative dead-ball evidence. Missing ball observations alone never imply D.

Run from the repository root:
    python team_model.py --out submission_artifacts/TEAM_NAME_PENDING_v2.csv
"""

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames


LABELS = ("H", "A", "D", "X")
HOME, AWAY, NOBODY = 0, 1, -1
FPS = 25.0

# Physical scales, not values selected against either public or private match labels.
LOCAL_SUPPORT_RADIUS_M = 4.0
NEAR_PLAYER_RADIUS_M = 4.5
NEAREST_MARGIN_SCALE_M = 4.0
MAX_PLAYER_SPEED_MPS = 12.0
MAX_BALL_SPEED_MPS = 40.0
STATIONARY_BALL_SPEED_MPS = 0.20
STATIONARY_BALL_STEP_M = 0.04
UNATTENDED_DISTANCE_M = 4.0
STATIONARY_DEAD_SECONDS = 1.0

# Temporal thresholds are expressed as evidence persistence in frames.
TEAM_EVIDENCE_THRESHOLD = 0.72
SWITCH_PERSISTENCE_FRAMES = 18  # 0.72 s of continuous opposing evidence.
INITIAL_PERSISTENCE_FRAMES = 5
RESTART_PERSISTENCE_FRAMES = 5
OUT_OF_FIELD_PERSISTENCE_FRAMES = 3
# Stationarity itself must already persist for one second in feature extraction; three further
# frames of the unattended condition confirm the state before decoding it as D.
STATIONARY_PERSISTENCE_FRAMES = 3
LIVE_UNCERTAINTY_GRACE_FRAMES = 12
DEAD_MISSING_GRACE_FRAMES = 30


def _parse_group(group):
    """Return x, y, speed or None for a missing ATD group."""
    if not group or group[0] == "N":
        return None
    values = group.split(",")
    if len(values) < 2:
        return None
    x, y = float(values[0]), float(values[1])
    speed = float(values[2]) if len(values) > 2 else float("nan")
    # The ATD uses literal NaN coordinates for an absent observation.  Treating the
    # parsed tuple as present made a NaN ball look visible and could poison all features.
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y, speed


def load_team_slots(metadata_path):
    """Read team names and per-slot team codes from the ATD XML."""
    root = ET.parse(metadata_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected exactly two Team entries in {}".format(metadata_path))
    team_ids = [team.get("id") for team in teams]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in {}".format(metadata_path))
    unknown = sorted(set(slots) - set(team_ids))
    if unknown:
        raise ValueError("track slot references unknown team IDs: {}".format(unknown))
    home_id = team_ids[0]
    team_codes = np.asarray([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)
    return tuple(team.findtext("Name") for team in teams), team_codes


def _team_weighted_mean(values, weights, mask):
    chosen = mask & np.isfinite(values) & (weights > 0)
    if not np.any(chosen):
        return 0.0
    return float(np.sum(values[chosen] * weights[chosen]) / np.sum(weights[chosen]))


def extract_features(raw_path, metadata_path, n_frames, pitch, half_blocks=None):
    """Stream ATD records into compact per-frame possession evidence features.

    Player slot motion is used only between adjacent records and is rejected when its derived
    speed exceeds a physically plausible bound, limiting the impact of slot identity switches.
    """
    team_names, slot_teams = load_team_slots(metadata_path)
    n_slots = len(slot_teams)
    pitch_length, pitch_width = pitch

    names = (
        "ball_visible", "ball_speed", "nearest_home", "nearest_away", "support_home",
        "support_away", "near_count_home", "near_count_away", "approach_home",
        "approach_away", "alignment_home", "alignment_away", "activity_home",
        "activity_away", "team_score", "evidence_valid", "stationary_unattended",
        "outside_pitch", "stationary_frames", "player_observed",
    )
    features = {key: np.zeros(n_frames, dtype=np.float32) for key in names}
    features["nearest_home"].fill(np.nan)
    features["nearest_away"].fill(np.nan)
    features["team_names"] = team_names

    prev_x = np.full(n_slots, np.nan, dtype=np.float64)
    prev_y = np.full(n_slots, np.nan, dtype=np.float64)
    prev_ball_x = prev_ball_y = float("nan")
    previous_ball_visible = False
    stationary_run = 0
    expected_frame_id = 1
    feature_reset_indices = set()
    if half_blocks is not None:
        for _half, first, last in half_blocks:
            feature_reset_indices.update((first, last + 1))

    with open(raw_path, "r") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > n_frames:
                raise ValueError("ATD has more than the expected {} records".format(n_frames))
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("malformed ATD record at line {}".format(line_number))
            frame_id = int(parts[0])
            if frame_id != expected_frame_id:
                raise ValueError("expected ATD frame_id {}, got {}".format(
                    expected_frame_id, frame_id))
            expected_frame_id += 1
            i = frame_id - 1
            if i in feature_reset_indices:
                prev_x.fill(np.nan)
                prev_y.fill(np.nan)
                prev_ball_x = prev_ball_y = float("nan")
                previous_ball_visible = False
                stationary_run = 0

            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError("frame {} has {} slots; metadata defines {}".format(
                    frame_id, len(groups), n_slots))
            px = np.full(n_slots, np.nan, dtype=np.float64)
            py = np.full(n_slots, np.nan, dtype=np.float64)
            ps = np.full(n_slots, np.nan, dtype=np.float64)
            for slot, group in enumerate(groups):
                parsed = _parse_group(group)
                if parsed is not None:
                    px[slot], py[slot], ps[slot] = parsed

            player_valid = np.isfinite(px) & np.isfinite(py)
            features["player_observed"][i] = float(np.count_nonzero(player_valid))
            ball = _parse_group(parts[2])
            features["ball_visible"][i] = float(ball is not None)
            current_ball_visible = ball is not None

            if not current_ball_visible:
                stationary_run = 0
                previous_ball_visible = False
                prev_x, prev_y = px, py
                continue

            bx, by, provided_ball_speed = ball
            ball_x_m, ball_y_m = bx * pitch_length, by * pitch_width
            dx = (px - bx) * pitch_length
            dy = (py - by) * pitch_width
            distances = np.sqrt(dx * dx + dy * dy)
            distances[~player_valid] = np.inf

            home_mask = (slot_teams == HOME) & player_valid
            away_mask = (slot_teams == AWAY) & player_valid
            home_distances = distances[home_mask]
            away_distances = distances[away_mask]
            home_slots = np.flatnonzero(home_mask)
            away_slots = np.flatnonzero(away_mask)
            nearest_home = float(np.min(home_distances)) if home_distances.size else float("inf")
            nearest_away = float(np.min(away_distances)) if away_distances.size else float("inf")
            features["nearest_home"][i] = nearest_home if math.isfinite(nearest_home) else np.nan
            features["nearest_away"][i] = nearest_away if math.isfinite(nearest_away) else np.nan

            weights = np.exp(-distances / LOCAL_SUPPORT_RADIUS_M)
            weights[~player_valid] = 0.0
            home_weights = weights[home_mask]
            away_weights = weights[away_mask]
            support_home = float(np.sum(home_weights))
            support_away = float(np.sum(away_weights))
            near_home = int(np.count_nonzero(home_distances <= NEAR_PLAYER_RADIUS_M))
            near_away = int(np.count_nonzero(away_distances <= NEAR_PLAYER_RADIUS_M))
            features["support_home"][i] = support_home
            features["support_away"][i] = support_away
            features["near_count_home"][i] = near_home
            features["near_count_away"][i] = near_away

            approach_by_slot = np.zeros(n_slots, dtype=np.float64)
            alignment_by_slot = np.zeros(n_slots, dtype=np.float64)
            alignment_valid = np.zeros(n_slots, dtype=bool)
            ball_displacement = float("nan")
            measured_ball_speed = provided_ball_speed
            if previous_ball_visible and np.isfinite(prev_ball_x) and np.isfinite(prev_ball_y):
                bdx = (bx - prev_ball_x) * pitch_length
                bdy = (by - prev_ball_y) * pitch_width
                ball_displacement = math.hypot(bdx, bdy)
                derived_speed = ball_displacement * FPS
                if not math.isfinite(measured_ball_speed) or measured_ball_speed < 0:
                    measured_ball_speed = derived_speed
                if derived_speed <= MAX_BALL_SPEED_MPS and ball_displacement > 1e-5:
                    player_continuity = player_valid & np.isfinite(prev_x) & np.isfinite(prev_y)
                    pdx = (px - prev_x) * pitch_length
                    pdy = (py - prev_y) * pitch_width
                    player_speed = np.sqrt(pdx * pdx + pdy * pdy) * FPS
                    continuity = player_continuity & (player_speed <= MAX_PLAYER_SPEED_MPS)
                    prev_distances = np.sqrt(
                        ((prev_x - prev_ball_x) * pitch_length) ** 2
                        + ((prev_y - prev_ball_y) * pitch_width) ** 2)
                    current_distances = distances
                    approach_by_slot[continuity] = np.clip(
                        (prev_distances[continuity] - current_distances[continuity]) * FPS,
                        -MAX_PLAYER_SPEED_MPS, MAX_PLAYER_SPEED_MPS)
                    ball_norm = math.hypot(bdx, bdy)
                    player_norm = np.sqrt(pdx * pdx + pdy * pdy)
                    moving = continuity & (player_norm > 1e-4)
                    alignment_by_slot[moving] = (
                        pdx[moving] * bdx + pdy[moving] * bdy
                    ) / (player_norm[moving] * ball_norm)
                    alignment_valid[moving] = True

            if not math.isfinite(measured_ball_speed) or measured_ball_speed < 0:
                measured_ball_speed = 0.0
            measured_ball_speed = min(float(measured_ball_speed), MAX_BALL_SPEED_MPS)
            features["ball_speed"][i] = measured_ball_speed

            features["approach_home"][i] = float(
                approach_by_slot[home_slots[np.argmin(distances[home_slots])]]
                if home_slots.size else 0.0)
            features["approach_away"][i] = float(
                approach_by_slot[away_slots[np.argmin(distances[away_slots])]]
                if away_slots.size else 0.0)
            align_weights = np.exp(-distances / 8.0)
            features["alignment_home"][i] = _team_weighted_mean(
                alignment_by_slot, align_weights, alignment_valid & (slot_teams == HOME))
            features["alignment_away"][i] = _team_weighted_mean(
                alignment_by_slot, align_weights, alignment_valid & (slot_teams == AWAY))

            speed_fraction = np.clip(np.nan_to_num(ps, nan=0.0) / 5.0, 0.0, 1.0)
            activity = weights * speed_fraction
            features["activity_home"][i] = float(np.sum(activity[home_mask]))
            features["activity_away"][i] = float(np.sum(activity[away_mask]))

            both_teams_present = math.isfinite(nearest_home) and math.isfinite(nearest_away)
            features["evidence_valid"][i] = float(np.any(player_valid))
            if both_teams_present:
                distance_margin = math.tanh(
                    (nearest_away - nearest_home) / NEAREST_MARGIN_SCALE_M)
            elif math.isfinite(nearest_home):
                distance_margin = 0.25
            elif math.isfinite(nearest_away):
                distance_margin = -0.25
            else:
                distance_margin = 0.0
            support_margin = (support_home - support_away) / (
                support_home + support_away + 1e-6)
            count_margin = (near_home - near_away) / float(near_home + near_away + 1)
            approach_margin = math.tanh(
                (features["approach_home"][i] - features["approach_away"][i]) / 3.0)
            alignment_margin = float(np.clip(
                features["alignment_home"][i] - features["alignment_away"][i], -1.0, 1.0))
            activity_home = float(features["activity_home"][i])
            activity_away = float(features["activity_away"][i])
            activity_margin = (activity_home - activity_away) / (
                activity_home + activity_away + 1e-6)
            score = (1.30 * distance_margin + 0.55 * support_margin
                     + 0.25 * count_margin + 0.35 * approach_margin
                     + 0.25 * alignment_margin + 0.15 * activity_margin)
            if not both_teams_present:
                # A one-sided tracking fragment is weak evidence, even when its only player is near.
                score = float(np.clip(score, -0.45, 0.45))
            features["team_score"][i] = float(np.clip(score, -2.85, 2.85))

            nearest_any = min(nearest_home, nearest_away)
            if previous_ball_visible and np.isfinite(ball_displacement):
                if (ball_displacement <= STATIONARY_BALL_STEP_M
                        and measured_ball_speed <= STATIONARY_BALL_SPEED_MPS):
                    stationary_run += 1
                else:
                    stationary_run = 0
            else:
                stationary_run = 0
            features["stationary_frames"][i] = stationary_run
            features["stationary_unattended"][i] = float(
                stationary_run >= STATIONARY_PERSISTENCE_FRAMES
                and nearest_any > UNATTENDED_DISTANCE_M)
            features["outside_pitch"][i] = float(
                bx < -0.01 or bx > 1.01 or by < -0.01 or by > 1.01)

            prev_x, prev_y = px, py
            prev_ball_x, prev_ball_y = bx, by
            previous_ball_visible = True

    if expected_frame_id != n_frames + 1:
        raise ValueError("ATD ended after {} records; expected {}".format(
            expected_frame_id - 1, n_frames))
    features["team_names"] = team_names
    return features


def decode_states(features, half_blocks=None):
    """Apply a persistent state decoder; weak/no observations produce X, not D."""
    n = len(features["team_score"])
    output = np.full(n, "X", dtype="<U1")
    state = "X"
    pending_state = None
    pending_start = None
    pending_count = 0
    uncertain_run = 0
    missing_run = 0
    half_starts = ({first for _half, first, _last in half_blocks}
                   if half_blocks is not None else set())
    graded_frames = None
    if half_blocks is not None:
        graded_frames = np.zeros(n, dtype=bool)
        for _half, first, last in half_blocks:
            graded_frames[max(0, first):min(n, last + 1)] = True

    for i in range(n):
        # Official per-half alignment defines the graded video blocks. Reset every
        # decoder memory at a half boundary and keep ungraded gaps neutral so no
        # possession or pending transition can leak over halftime.
        if i in half_starts:
            state = "X"
            pending_state, pending_start, pending_count = None, None, 0
            uncertain_run = missing_run = 0
        if graded_frames is not None and not graded_frames[i]:
            output[i] = "X"
            continue
        ball_visible = bool(features["ball_visible"][i])
        score = float(features["team_score"][i])
        evidence_valid = bool(features["evidence_valid"][i])
        outside = bool(features["outside_pitch"][i])
        stationary_unattended = bool(features["stationary_unattended"][i])

        dead_target = None
        if outside:
            dead_target = ("D", OUT_OF_FIELD_PERSISTENCE_FRAMES)
        elif stationary_unattended:
            dead_target = ("D", STATIONARY_PERSISTENCE_FRAMES)

        if dead_target is not None:
            target, persistence = dead_target
            missing_run = 0
            uncertain_run = 0
            if state == target:
                output[i] = "D"
                pending_state, pending_start, pending_count = None, None, 0
            else:
                if pending_state != target:
                    pending_state, pending_start, pending_count = target, i, 1
                else:
                    pending_count += 1
                if pending_count >= persistence:
                    output[pending_start:i + 1] = "D"
                    state = "D"
                    pending_state, pending_start, pending_count = None, None, 0
                elif state in ("H", "A"):
                    output[i] = state
                else:
                    output[i] = "X"
            continue

        if state == "D" and not ball_visible:
            missing_run += 1
            output[i] = "D" if missing_run <= DEAD_MISSING_GRACE_FRAMES else "X"
            continue
        missing_run = 0

        target = None
        if ball_visible and evidence_valid and abs(score) >= TEAM_EVIDENCE_THRESHOLD:
            target = "H" if score > 0 else "A"

        if target is None:
            uncertain_run += 1
            pending_state, pending_start, pending_count = None, None, 0
            if state in ("H", "A") and uncertain_run <= LIVE_UNCERTAINTY_GRACE_FRAMES:
                output[i] = state
            elif state == "D" and ball_visible:
                output[i] = "X"
                state = "X"
            else:
                output[i] = "X"
            continue

        uncertain_run = 0
        if target == state:
            output[i] = state
            pending_state, pending_start, pending_count = None, None, 0
            continue

        if pending_state != target:
            pending_state, pending_start, pending_count = target, i, 1
        else:
            pending_count += 1

        if state == "X":
            persistence = INITIAL_PERSISTENCE_FRAMES
        elif state == "D":
            persistence = RESTART_PERSISTENCE_FRAMES
        else:
            persistence = SWITCH_PERSISTENCE_FRAMES

        if pending_count >= persistence:
            output[pending_start:i + 1] = target
            state = target
            pending_state, pending_start, pending_count = None, None, 0
        elif state in ("H", "A"):
            output[i] = state
        else:
            output[i] = "X"

    return output


def write_submission(path, labels):
    """Write the strict official frame,label format after basic internal checks."""
    if labels.ndim != 1 or not np.all(np.isin(labels, LABELS)):
        raise ValueError("labels must be a one-dimensional array containing only {}".format(LABELS))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        for frame, label in enumerate(labels):
            handle.write("{},{}\n".format(frame, label))
    return path


def _duration_summary(lengths):
    values = np.asarray(lengths, dtype=np.float64) / FPS
    return {
        "count": int(values.size),
        "mean_seconds": float(np.mean(values)) if values.size else 0.0,
        "median_seconds": float(np.median(values)) if values.size else 0.0,
        "p90_seconds": float(np.percentile(values, 90)) if values.size else 0.0,
    }


def _runs(labels, target):
    lengths, current = [], 0
    for label in labels:
        if label == target:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def diagnostics(features, labels, extraction_seconds, decoding_seconds, total_seconds,
                half_blocks=None):
    """Label-free operational and segmentation diagnostics."""
    counts = {label: int(np.count_nonzero(labels == label)) for label in LABELS}
    possession_runs = []
    start = None
    active_label = None
    for i, label in enumerate(labels):
        if label in ("H", "A"):
            if start is None or label != active_label:
                if start is not None:
                    possession_runs.append(i - start)
                start, active_label = i, label
        elif start is not None:
            possession_runs.append(i - start)
            start, active_label = None, None
    if start is not None:
        possession_runs.append(len(labels) - start)

    adjacent_labeled = (labels[1:] != "X") & (labels[:-1] != "X")
    transitions = int(np.count_nonzero(
        adjacent_labeled & (labels[1:] != labels[:-1]))) if len(labels) > 1 else 0
    missing = features["ball_visible"] == 0
    missing_output_counts = {label: int(np.count_nonzero(labels[missing] == label))
                             for label in LABELS}
    durations = np.asarray(possession_runs, dtype=np.float64) / FPS
    x_visible = int(np.count_nonzero((labels == "X") & ~missing))
    x_missing = int(np.count_nonzero((labels == "X") & missing))
    result = {
        "frames": len(labels),
        "label_counts": counts,
        "predicted_possession_spells": len(possession_runs),
        "spell_duration_seconds": _duration_summary(possession_runs),
        "ball_observed_frames": int(np.count_nonzero(~missing)),
        "ball_observed_percent": float(100.0 * np.mean(~missing)),
        "ball_missing_frames": int(np.count_nonzero(missing)),
        "labels_on_missing_ball_frames": missing_output_counts,
        "X_on_ball_visible_frames": x_visible,
        "X_on_ball_missing_frames": x_missing,
        "D_run_duration_seconds": _duration_summary(_runs(labels, "D")),
        "X_run_duration_seconds": _duration_summary(_runs(labels, "X")),
        "stationary_unattended_evidence_frames": int(np.count_nonzero(
            features["stationary_unattended"])),
        "outside_pitch_evidence_frames": int(np.count_nonzero(features["outside_pitch"])),
        "adjacent_labeled_pairs": int(np.count_nonzero(adjacent_labeled)),
        "transition_count_between_adjacent_non_X_frames": transitions,
        "H_to_A_adjacent_non_X": int(np.count_nonzero(
            adjacent_labeled & (labels[:-1] == "H") & (labels[1:] == "A"))),
        "A_to_H_adjacent_non_X": int(np.count_nonzero(
            adjacent_labeled & (labels[:-1] == "A") & (labels[1:] == "H"))),
        "extraction_seconds": float(extraction_seconds),
        "decoding_seconds": float(decoding_seconds),
        "total_model_seconds": float(total_seconds),
    }
    if half_blocks is not None:
        result["frame_counts_by_half_and_ungraded"] = {}
        graded = np.zeros(len(labels), dtype=bool)
        for half, first, last in half_blocks:
            start, stop = max(0, first), min(len(labels), last + 1)
            graded[start:stop] = True
            block = labels[start:stop]
            result["frame_counts_by_half_and_ungraded"][str(half)] = {
                label: int(np.count_nonzero(block == label)) for label in LABELS
            }
        ungraded = labels[~graded]
        result["frame_counts_by_half_and_ungraded"]["ungraded"] = {
            label: int(np.count_nonzero(ungraded == label)) for label in LABELS
        }
        result["spell_duration_seconds_by_half"] = {
            str(half): _duration_summary(_runs(labels[max(0, first):min(len(labels), last + 1)],
                                                "H") +
                                         _runs(labels[max(0, first):min(len(labels), last + 1)],
                                                "A"))
            for half, first, last in half_blocks
        }
        result["D_run_duration_seconds_by_region"] = {}
        result["X_run_duration_seconds_by_region"] = {}
        for half, first, last in half_blocks:
            block = labels[max(0, first):min(len(labels), last + 1)]
            result["D_run_duration_seconds_by_region"][str(half)] = _duration_summary(
                _runs(block, "D"))
            result["X_run_duration_seconds_by_region"][str(half)] = _duration_summary(
                _runs(block, "X"))
        ungraded_runs = []
        for start, stop in ((0, max(0, half_blocks[0][1])),
                            (half_blocks[0][2] + 1, half_blocks[1][1]),
                            (half_blocks[1][2] + 1, len(labels))):
            ungraded_runs.extend(_runs(labels[max(0, start):min(len(labels), stop)], "X"))
        result["D_run_duration_seconds_by_region"]["ungraded"] = _duration_summary([])
        result["X_run_duration_seconds_by_region"]["ungraded"] = _duration_summary(
            ungraded_runs)
    return result


def generate(data_dir=None, out_path=None):
    """Run the V2 model on the full tactical match and emit label-free diagnostics."""
    metadata_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    total_start = time.perf_counter()
    extraction_start = time.perf_counter()
    half_blocks = half_video_blocks(GAME)
    features = extract_features(raw_path, metadata_path, n_frames, GAMES[GAME]["pitch"],
                                half_blocks)
    extraction_seconds = time.perf_counter() - extraction_start
    decoding_start = time.perf_counter()
    labels = decode_states(features, half_video_blocks(GAME))
    decoding_seconds = time.perf_counter() - decoding_start
    if len(labels) != n_frames:
        raise ValueError("model produced {} frames; expected {}".format(len(labels), n_frames))
    if not np.all(np.isin(labels, LABELS)):
        raise ValueError("model emitted an invalid label")
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_artifacts",
                                "TEAM_NAME_PENDING_v2.csv")
    write_submission(out_path, labels)
    total_seconds = time.perf_counter() - total_start
    report = diagnostics(features, labels, extraction_seconds, decoding_seconds, total_seconds,
                         half_blocks)
    report["output"] = os.path.abspath(out_path)
    report["home_team"] = features["team_names"][0]
    report["away_team"] = features["team_names"][1]
    report["smart_tagging_used"] = False
    report["homography_used"] = False
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="data bundle root (or MTAH_DATA_DIR)")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args(argv)
    report = generate(args.data_dir, args.out)
    print("V2 model diagnostics (no ground-truth scoring)")
    for key, value in report.items():
        print("  {}: {}".format(key, value))


if __name__ == "__main__":
    main()
