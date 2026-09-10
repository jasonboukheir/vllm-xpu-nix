"""Join captured native dispatch to offline compiler resource metadata."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codegen", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--cxxfilt", default="c++filt")
    parser.add_argument("--candidate-q", type=int, default=128)
    parser.add_argument("--candidate-subgroups", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    codegen = json.loads(args.codegen.read_text())
    gates = json.loads((args.gates / "report.json").read_text())
    if gates["status"] not in ("gates-passed", "measured"):
        raise ValueError("correctness gates must pass before dispatch attestation")
    records = []
    for library in codegen["libraries"]:
        arm = library["label"]
        names = "\n".join(k["kernel_name"] for k in library["kernels"])
        demangled = subprocess.check_output(
            [args.cxxfilt], input=names, text=True
        ).splitlines()
        lookup = {
            name.removeprefix("typeinfo name for "): kernel
            for name, kernel in zip(demangled, library["kernels"], strict=True)
        }
        identity = next(i for i in gates["identities"] if i["arm"] == arm)
        if identity["sha256"] != library["sha256"]:
            raise ValueError("offline library differs from mapped gate library")
        for gate in (g for g in gates["gates"] if g["arm"] == arm):
            trace = args.gates / f"trace-{arm}-{gate['case']}.json"
            events = json.loads(trace.read_text())["traceEvents"]
            native = [
                e
                for e in events
                if e.get("cat") == "kernel" and "XeFMHAFwdKernel" in e.get("name", "")
            ]
            if len(native) != 1 or native[0]["name"] not in lookup:
                raise ValueError(f"ambiguous or unknown prefill dispatch in {trace}")
            kernel = lookup[native[0]["name"]]
            q, subgroups = (
                (args.candidate_q, args.candidate_subgroups)
                if arm == "candidate"
                else (256, 32)
            )
            if (
                f"cute::Layout<cute::C<{q}>, cute::C<1> >" not in native[0]["name"]
                or f"cute::tuple<cute::C<{subgroups}>, cute::C<1>, cute::C<1> >"
                not in native[0]["name"]
            ):
                raise ValueError("trace does not select intended Q/subgroup policy")
            records.append(
                {
                    "arm": arm,
                    "case": gate["case"],
                    "trace": str(trace),
                    "query_rows_per_workgroup": q,
                    "subgroups_per_workgroup": subgroups,
                    "query_rows_per_subgroup": q // subgroups,
                    "threads_per_workgroup": subgroups * 16,
                    "library_sha256": library["sha256"],
                    **kernel,
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "method": "Exact demangled device-trace name joined to .ze_info/.text from the same mapped immutable DSO",
                "limits": "No spill entry means none emitted; GRF/SLM metadata is not measured occupancy or traffic. Tile and subgroup dimensions are checked against the actual generated template name.",
                "dispatch": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
