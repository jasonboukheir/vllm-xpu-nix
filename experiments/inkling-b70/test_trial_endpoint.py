"""Check the cache-reset clients' model identity guard using a mock model list."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

from offload_replay import MODEL_ID, MODEL_REVISION, SNAPSHOT_ROOT, Trial


class TrialEndpointTest(TestCase):
    def trial(self, alias="sunny-chat", port=18001):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Trial(
            SimpleNamespace(model=alias, port=port, output=Path(directory.name) / "run")
        )

    def test_two_aliases_and_model_roots(self):
        for alias in ("inkling-trial", "sunny-chat"):
            for root in (MODEL_ID, SNAPSHOT_ROOT):
                with self.subTest(alias=alias, root=root):
                    trial = self.trial(alias)
                    model = {"id": alias, "root": root}
                    trial.request = Mock(return_value={"data": [model]})
                    self.assertEqual(trial.verify_model(), model)
                    self.assertEqual(trial.doc["models"], [model])
                    trial.request.assert_called_once_with("/v1/models")

    def test_mismatch_stops_before_reset(self):
        valid = {"id": "sunny-chat", "root": MODEL_ID}
        for rows in (
            [],
            [valid, valid],
            [dict(valid, id="inkling-trial")],
            [{"id": "sunny-chat"}],
            [dict(valid, root="RedHatAI/Qwen3.8-27B-INT4")],
            [dict(valid, root=SNAPSHOT_ROOT.replace(MODEL_REVISION, "main"))],
            [dict(valid, revision="other")],
        ):
            with self.subTest(rows=rows):
                trial = self.trial()
                trial.request = Mock(return_value={"data": rows})
                with self.assertRaises(ValueError):
                    trial.run()
                trial.request.assert_called_once_with("/v1/models")

    def test_production_port_rejected_before_requests(self):
        with self.assertRaisesRegex(ValueError, "isolated loopback trial"):
            self.trial(port=8000)

    def test_long_replay_requires_advertised_room_before_reset(self):
        for limit in (None, 245791, "262144"):
            with self.subTest(limit=limit):
                trial = self.trial()
                trial.args.depth = 245760
                model = {
                    "id": trial.args.model,
                    "root": MODEL_ID,
                    "max_model_len": limit,
                }
                trial.request = Mock(return_value={"data": [model]})
                with self.assertRaisesRegex(ValueError, "at least 245792"):
                    trial.run()
                trial.request.assert_called_once_with("/v1/models")

    def test_long_replay_at_advertised_boundary(self):
        trial = self.trial()
        trial.args.depth = 245760
        model = {"id": trial.args.model, "root": MODEL_ID, "max_model_len": 245792}
        trial.request = Mock(return_value={"data": [model]})
        trial.deep_replay, trial.reset, trial.stats, trial.finish = (
            Mock() for _ in range(4)
        )
        trial.run()
        trial.deep_replay.assert_called_once_with(245760)


if __name__ == "__main__":
    main()
