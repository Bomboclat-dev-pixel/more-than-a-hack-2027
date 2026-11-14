import os
import tempfile
import unittest

import numpy as np

import team_model as model


def _features(scores, visible=None, valid=None, outside=None, stationary=None):
    n = len(scores)
    return {
        "team_score": np.asarray(scores, dtype=np.float32),
        "ball_visible": np.asarray(visible if visible is not None else [1] * n,
                                    dtype=np.float32),
        "evidence_valid": np.asarray(valid if valid is not None else [1] * n,
                                      dtype=np.float32),
        "outside_pitch": np.asarray(outside if outside is not None else [0] * n,
                                    dtype=np.float32),
        "stationary_unattended": np.asarray(
            stationary if stationary is not None else [0] * n, dtype=np.float32),
    }


class TeamModelTests(unittest.TestCase):
    def test_missing_ball_is_not_dead_ball(self):
        labels = model.decode_states(_features(
            [0] * 20, visible=[0] * 20, valid=[0] * 20))
        self.assertTrue(np.all(labels == "X"))
        self.assertFalse(np.any(labels == "D"))

    def test_hysteresis_rejects_short_opponent_burst_and_accepts_persistent_switch(self):
        scores = ([1.2] * 10) + ([-1.2] * 17) + ([1.2] * 5) + ([-1.2] * 20)
        labels = model.decode_states(_features(scores))
        self.assertTrue(np.all(labels[:32] == "H"))
        self.assertTrue(np.all(labels[32:] == "A"))

    def test_dead_ball_requires_persistent_positive_evidence(self):
        outside = [0, 0, 0, 1, 1, 1, 0, 0]
        labels = model.decode_states(_features(
            [0] * len(outside), outside=outside))
        self.assertEqual(list(labels), ["X", "X", "X", "D", "D", "D", "X", "X"])

    def test_atd_feature_extraction_handles_missing_observations(self):
        xml = """<main><Teams>
          <Team id="H"><Name>Home</Name></Team><Team id="A"><Name>Away</Name></Team>
          </Teams><Players><Player teamId="H"/><Player teamId="A"/></Players></main>"""
        rows = [
            "1:0.50,0.50,1.0;0.80,0.50,2.0:0.51,0.50,0.5",
            "2:0.50,0.50,1.0;NaN,NaN,NaN:NaN,NaN,NaN",
            "3:0.52,0.50,1.0;0.80,0.50,2.0:0.53,0.50,0.5",
        ]
        with tempfile.TemporaryDirectory() as folder:
            xml_path = os.path.join(folder, "meta.xml")
            raw_path = os.path.join(folder, "tracking.txt")
            with open(xml_path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")
            features = model.extract_features(raw_path, xml_path, 3, (105.3, 68.0))

        self.assertEqual(features["team_names"], ("Home", "Away"))
        self.assertEqual(features["ball_visible"].tolist(), [1.0, 0.0, 1.0])
        self.assertGreater(features["nearest_home"][0], 0.0)
        self.assertGreater(features["nearest_away"][0], features["nearest_home"][0])
        self.assertEqual(features["player_observed"][1], 1.0)
        self.assertEqual(features["ball_speed"][1], 0.0)
        self.assertEqual(features["player_observed"][1], 1.0)

    def test_nonfinite_ball_coordinates_are_missing(self):
        xml = """<main><Teams><Team id='H'><Name>Home</Name></Team>
          <Team id='A'><Name>Away</Name></Team></Teams>
          <Players><Player teamId='H'/><Player teamId='A'/></Players></main>"""
        row = "1:0.50,0.50,1.0;0.80,0.50,2.0:NaN,NaN,NaN"
        with tempfile.TemporaryDirectory() as folder:
            xml_path = os.path.join(folder, "meta.xml")
            raw_path = os.path.join(folder, "tracking.txt")
            with open(xml_path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write(row + "\n")
            features = model.extract_features(raw_path, xml_path, 1, (105.3, 68.0))
        self.assertEqual(features["ball_visible"].tolist(), [0.0])
        self.assertTrue(np.isnan(features["nearest_home"][0]))

    def test_feature_motion_history_resets_at_half_boundary(self):
        xml = """<main><Teams><Team id='H'><Name>Home</Name></Team>
          <Team id='A'><Name>Away</Name></Team></Teams>
          <Players><Player teamId='H'/><Player teamId='A'/></Players></main>"""
        rows = [f"{i}:0.50,0.50,0.0;0.80,0.50,0.0:0.51,0.50,0.0"
                for i in range(1, 5)]
        with tempfile.TemporaryDirectory() as folder:
            xml_path = os.path.join(folder, "meta.xml")
            raw_path = os.path.join(folder, "tracking.txt")
            with open(xml_path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")
            features = model.extract_features(
                raw_path, xml_path, 4, (105.3, 68.0), [(1, 0, 2), (2, 3, 3)])
        self.assertEqual(features["stationary_frames"][3], 0.0)

    def test_decoder_resets_across_ungraded_gap_and_second_half(self):
        features = _features([1.2] * 20 + [0.0] * 4 + [-1.2] * 16)
        labels = model.decode_states(features, half_blocks=[(1, 0, 9), (2, 20, 39)])
        self.assertTrue(np.all(labels[:10] == "H"))
        self.assertTrue(np.all(labels[10:20] == "X"))
        self.assertTrue(np.all(labels[20:24] == "X"))
        self.assertTrue(np.all(labels[24:] == "A"))

    def test_submission_writer_emits_exact_frame_labels(self):
        labels = np.asarray(["H", "D", "X", "A"])
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "candidate.csv")
            model.write_submission(output, labels)
            with open(output, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines, ["frame,label", "0,H", "1,D", "2,X", "3,A"])


if __name__ == "__main__":
    unittest.main()
