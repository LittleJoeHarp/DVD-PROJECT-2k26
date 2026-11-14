"""Ordered project entry point; run from any working directory."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

from common import Context, PROCESS, ROOT, save_json, sha256


def setup(ctx):
    canonical = ROOT / "models" / ctx.config["model_file"]
    if (ROOT / "model.tar").is_file():
        with tarfile.open(ROOT / "model.tar") as archive:
            original = archive.extractfile("model/32nm_MGK.pm").read()
        source = "model.tar:model/32nm_MGK.pm"
    elif canonical.is_file():
        original = canonical.read_bytes()
        source = "models/32nm_MGK.pm (canonical instructor-supplied card)"
    else:
        raise FileNotFoundError("The submission needs models/32nm_MGK.pm")
    duplicate = None
    if (ROOT / "models.zip").is_file():
        with zipfile.ZipFile(ROOT / "models.zip") as archive:
            duplicate = archive.read("models/32nm_MGK.pm")
        if original.strip() != duplicate.strip():
            raise RuntimeError("Assigned model copies differ substantively")
    if canonical.is_file() and canonical.read_bytes() != original:
        raise RuntimeError("Canonical model differs from the original assigned card")
    path = ctx.models / ctx.config["model_file"]
    path.write_bytes(original)
    parts = re.split(r"(?im)^\.model\s+", original.decode())[1:]
    nominal = {}
    card_nominal = {}
    for part in parts:
        name = part.split()[0].lower()
        params = dict((key.lower(), float(value)) for key, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)", part))
        card_nominal[name] = {key: params[key] for key in PROCESS}
        nominal[name] = dict(card_nominal[name])
        nominal[name]["ndep"] *= 1e6  # cm^-3 in SPICE -> m^-3 in the CSV/features.
    assert set(nominal) == {"nmos", "pmos"}
    save_json(ctx.models / "metadata.json", {
        "source": source, "sha256": sha256(path),
        "duplicate_source": "models.zip:models/32nm_MGK.pm" if duplicate is not None else None,
        "duplicates_equal_after_strip": True if duplicate is not None else None, "original_preserved": True,
        "nominal": nominal, "card_nominal": card_nominal, "model_names": ["nmos", "pmos"],
        "ndep_units": "CSV/features: m^-3; divide by 1e6 at the SPICE boundary (card: cm^-3)",
        "oxide_warning": "Original PMOS toxe != toxp + dtox; ngspice ignores dtox. No card edits.",
    })
    save_json(ctx.results / "checks" / "configuration.json", ctx.config)
    from spice import smoke
    result = smoke(ctx)
    seconds = sum(r["execution"]["elapsed_s"] for r in result.values())
    uniform_points = 4*ctx.config["samples_per_case"]
    supplementary_points = 4*ctx.config.get("supplementary_samples_per_case", 0)
    ctx.check("workload", {
        "uniform_transistor_operating_points": uniform_points,
        "supplementary_transistor_operating_points": supplementary_points,
        "transistor_operating_points": uniform_points+supplementary_points, "gate_operating_points": 3600,
        "circuit_operating_points": 51200, "total_required_operating_points": uniform_points+supplementary_points+54800,
        "circuit_nand2_instances": 28, "circuit_mos_count": 112,
        "strategy": "250-row transistor batches; gate/circuit corner batches; four subprocess workers",
        "smoke_12_points_elapsed_s": seconds,
        "timing_note": "Startup-dominated smoke timing; benchmark each larger stage before its full run.",
        "simulator": subprocess.check_output(["ngspice", "--version"], text=True).strip(),
    })
    print("Part 1 passed: original model, four cases, process-change/reset checks.", flush=True)


def execute(ctx, stage):
    if stage == "setup":
        setup(ctx)
    elif stage == "data":
        from data import generate, recheck
        generate(ctx)
        recheck(ctx)
    elif stage == "train":
        from train import train
        train(ctx)
    elif stage == "gates":
        from gates import references, recheck
        references(ctx)
        recheck(ctx)
    elif stage == "predict":
        from gates import predictions
        predictions(ctx)
    elif stage == "circuit":
        from circuit import evaluate
        evaluate(ctx)

    elif stage == "benchmark":
        if not (ROOT / "baseline").is_dir() or not (ROOT / "src/benchmark.py").is_file():
            raise RuntimeError("Historical benchmarking is optional and needs the local baseline archive and benchmark.py")
        from benchmark import compare_baseline
        compare_baseline(ctx)


def main():
    stages = ["setup", "data", "train", "gates", "predict", "circuit"]
    if (ROOT / "baseline").is_dir() and (ROOT / "src/benchmark.py").is_file():
        stages.append("benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=list(dict.fromkeys([*stages, "benchmark", "all"])))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    ctx = Context(args.output)
    os.environ.setdefault("MPLCONFIGDIR", str(ctx.scratch / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    for stage in stages if args.stage == "all" else [args.stage]:
        execute(ctx, stage)


if __name__ == "__main__":
    main()
