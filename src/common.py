"""Shared paths, configuration, and reproducibility helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
PROCESS = ("vth0", "toxe", "ndep", "u0", "xj")
FEATURES = ("vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROCESS)
TARGETS = ("id", "ig", "is", "ib")
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"


def save_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Context:
    def __init__(self, output=ROOT):
        self.root = ROOT
        self.out = Path(output).resolve()
        self.config = json.loads((ROOT / "config.json").read_text())
        for name in ("models", "netlists", "data", "results", "scratch"):
            setattr(self, name, self.out / name)
            (self.out / name).mkdir(parents=True, exist_ok=True)
        for name in ("checks", "plots", "tables", "trained"):
            (self.results / name).mkdir(exist_ok=True)

    @property
    def nominal(self):
        return json.loads((self.models / "metadata.json").read_text())["nominal"]

    def check(self, name, value):
        save_json(self.results / "checks" / f"{name}.json", value)


def nominal_row(ctx, case):
    pmos = case.startswith("pmos")
    on = case.endswith("_on")
    return dict(vdd=1.0, temp=27.0, w=64e-9, vd=0.0 if pmos else 1.0,
                vs=1.0 if pmos else 0.0,
                vg=float((pmos and not on) or (not pmos and on)),
                vb=float(pmos), **ctx.nominal["pmos" if pmos else "nmos"])


def kcl_check(currents, config):
    values = np.asarray(currents)
    residual = np.abs(values.sum(axis=1))
    scale = np.abs(values).max(axis=1)
    absolute = config["kcl_absolute_tolerance_A"]
    limit = absolute + config["kcl_relative_tolerance"] * scale
    return {
        "maximum_residual_A": float(residual.max()),
        "median_residual_A": float(np.median(residual)),
        "p95_residual_A": float(np.percentile(residual, 95)),
        "absolute_tolerance_A": absolute,
        "absolute_only_exceedances": int((residual > absolute).sum()),
        "scale_aware_failures": int((residual > limit).sum()),
        "relative_tolerance": config["kcl_relative_tolerance"],
    }
