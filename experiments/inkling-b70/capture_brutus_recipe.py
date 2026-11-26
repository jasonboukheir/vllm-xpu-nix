"""Capture a realized Brutus unit and HF retention manifests without activating a generation."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial", type=Path, required=True, help="experimental flake checkout"
    )
    parser.add_argument(
        "--host", type=Path, required=True, help="Brutus host flake checkout"
    )
    parser.add_argument(
        "--phased",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the recipe's phasedDecode flag for this preview",
    )
    parser.add_argument(
        "--mixed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the recipe's mixedPrefill flag for this preview",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON receipt")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Specify a fresh output file: {args.output}")
    trial = args.trial.resolve(strict=True)
    host = args.host.resolve(strict=True)
    expr = Path(__file__).with_name("evaluate-brutus-recipe.nix").resolve()

    def revision(path):
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()

    cmd = [
        "nix",
        "build",
        "--no-link",
        "--print-out-paths",
        "--no-write-lock-file",
        "--impure",
        "--max-jobs",
        "1",
        "--cores",
        "2",
        "--file",
        str(expr),
        "--argstr",
        "trialPath",
        str(trial),
        "--argstr",
        "hostPath",
        str(host),
        "--argstr",
        "trialRev",
        revision(trial),
        "--argstr",
        "hostRev",
        revision(host),
    ]
    if args.phased is not None:
        cmd += ["--arg", "phasedDecode", "true" if args.phased else "false"]
    if args.mixed is not None:
        cmd += ["--arg", "mixedPrefill", "true" if args.mixed else "false"]
    preview_file = Path(subprocess.check_output(cmd, text=True).strip())
    record = json.loads(preview_file.read_text())
    unit = Path(record["unit"])
    lines = unit.read_text().splitlines()
    (argv,) = [
        shlex.split(line.removeprefix("ExecStart="))
        for line in lines
        if line.startswith("ExecStart=")
    ]
    unit_env = {}
    for line in lines:
        if line.startswith("Environment="):
            for assignment in shlex.split(line.removeprefix("Environment=")):
                key, value = assignment.split("=", 1)
                unit_env[key] = value
    if any(unit_env.get(k) != v for k, v in record["environment"].items()):
        raise ValueError(
            "Realized service environment differs from its selected recipe"
        )
    if argv != shlex.split(record.pop("command")):
        raise ValueError("Generated unit differs from the evaluated service command")
    if argv[:3] != [record["runtime"] + "/bin/vllm", "serve", record["model"]]:
        raise ValueError("Unexpected executable or checkpoint in the Brutus command")
    flags = [a for a in argv if a.startswith("--")]
    if len(flags) != len(set(flags)) or "--chat-template" in flags:
        raise ValueError(
            "Unexpected repeated option or chat template in the Brutus command"
        )
    if (
        argv[argv.index("--revision") + 1] != record["revision"]
        or json.loads(argv[argv.index("--speculative-config") + 1])["revision"]
        != record["revision"]
    ):
        raise ValueError(
            "Target and bundled draft must use the pinned checkpoint revision"
        )
    if args.phased is not None and record["environment"][
        "INK2_KVARN_PHASED_DECODE"
    ] != str(int(args.phased)):
        raise ValueError("Unexpected phased reader setting")
    if args.mixed is not None and record["environment"][
        "INK2_KVARN_MIXED_PREFILL"
    ] != str(int(args.mixed)):
        raise ValueError("Unexpected mixed prefill setting")
    for name, source in record["manifests"].copy().items():
        contents = json.loads(Path(source).read_text())
        root = {
            "source": "vllm-xpu.chat",
            "type": "model",
            "repo": record["model"],
            "revision": record["revision"],
        }
        if root not in contents["roots"]:
            raise ValueError(f"Pinned model missing from {name} retention manifest")
        record["manifests"][name] = {"file": source, "contents": contents}
    record.update(argv=argv, preview_file=str(preview_file), activated=False)
    with args.output.open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
