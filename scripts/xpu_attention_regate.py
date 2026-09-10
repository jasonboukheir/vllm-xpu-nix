"""Gate a fresh attention DSO against retained, verified original references."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.xpu_attention_screen import Worker


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    if prior["status"] != "gates-passed":
        raise ValueError("original gate run is incomplete")
    originals = [g for g in prior["gates"] if g["arm"] == "original"]
    controls = [g for g in prior["gates"] if g["arm"] == "control"]
    if not originals or not all(g["pass"] for g in originals + controls):
        raise ValueError("original/control gates failed")
    if {(g["manifest"], g["index"]) for g in originals} != {
        (g["manifest"], g["index"]) for g in controls
    }:
        raise ValueError("original/control cases differ")
    identities = [i for i in prior["identities"] if i["arm"] in ("original", "control")]
    for identity in identities:
        if (
            digest(identity["library"]) != identity["sha256"]
            or digest(identity["binding"]) != identity["binding_sha256"]
        ):
            raise ValueError("retained original/control library changed")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output)
    for name in ("xpu_attention_worker.py", "xpu_attention_screen.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output)
    report = {
        "status": "running",
        "prior": str(args.prior.resolve()),
        "prior_sha256": digest(args.prior),
        "gates": originals + controls,
        "identities": identities,
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    worker = Worker(args.python, args.candidate, args.output, "candidate")
    report["identities"].append(
        {"phase": "gate", "arm": "candidate", **worker.identity}
    )
    try:
        for gate in originals:
            case = gate["case"]
            reference = args.prior.parent / f"reference-{case}.pt"
            if digest(reference) != gate["reference_sha256"]:
                raise ValueError("original reference checksum mismatch")
            worker.call(
                action="prepare", manifest=gate["manifest"], index=gate["index"]
            )
            result = worker.call(
                action="validate",
                reference=str(reference.resolve()),
                make_reference=False,
            )
            report["gates"].append({**gate, **result, "arm": "candidate"})
            save()
            if not result["pass"]:
                raise ValueError(f"candidate failed correctness: {case}")
            worker.call(
                action="trace",
                path=str((args.output / f"trace-candidate-{case}.json").resolve()),
            )
            print(json.dumps({"case": case, **result}), flush=True)
        report["status"] = "gates-passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        worker.close()
        save()


if __name__ == "__main__":
    main()
