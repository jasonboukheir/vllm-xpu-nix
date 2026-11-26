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


if __name__ == "__main__":
    main()
