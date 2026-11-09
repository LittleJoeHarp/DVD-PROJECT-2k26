"""NGSpice operating-point batches with explicit row IDs and retained evidence."""
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from common import OPTIONS, PROCESS, kcl_check, nominal_row, save_json


def number(value):
    return f"{float(value):.17g}"


def include_path(ctx, directory):
    return os.path.relpath(ctx.models / ctx.config["model_file"], directory)


def single_deck(ctx, case, directory, row=None):
    row = nominal_row(ctx, case) if row is None else row
    polarity = case.split("_")[0]
    lines = [f"Single transistor {case}",
             f'.include "{include_path(ctx, directory)}"', OPTIONS,
             f'.temp {number(row["temp"])}',
             f'M1 d g s b {polarity} W={number(row["w"])} L={number(ctx.config["length_m"])}']
    lines += [f"V{terminal} {terminal} 0 {number(row['v' + terminal])}" for terminal in "dgsb"]
    return "\n".join(lines)


def process_commands(polarity, values):
    return [f"altermod {polarity} {name} = {number(values[name] / (1e6 if name == 'ndep' else 1))}"
            for name in PROCESS]


def single_commands(case, row):
    # Reset before altering, so BSIM geometry/temperature caches cannot carry across rows.
    return ["reset", *process_commands(case.split("_")[0], row),
            f'alter m1 w = {number(row["w"])}',
            f'set temp = {number(row["temp"])}',
            *[f"alter v{t} = {number(row['v'+t])}" for t in "dgsb"]]


def run_batch(ctx, tag, deck, commands, expressions, prefix=()):
    directory = ctx.scratch / tag
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "values.dat"
    output.unlink(missing_ok=True)
    control = [".control", "set noaskquit", "set numdgt=17", "set wr_singlescale",
               "set appendwrite", *prefix]
    for index, changes in enumerate(commands):
        control.extend(changes)
        control += ["op", f"let row_id = {index}", "setscale row_id",
                    "wrdata values.dat " + " ".join(expressions), "destroy $curplot"]
    control += ["quit 0", ".endc", ".end"]
    filename = directory / "batch.cir"
    filename.write_text(deck + "\n" + "\n".join(control) + "\n")
    start = time.perf_counter()
    result = subprocess.run(["ngspice", "-b", filename.name], cwd=directory,
                            capture_output=True, text=True, timeout=900)
    elapsed = time.perf_counter() - start
    log = result.stdout + "\n" + result.stderr
    (directory / "ngspice.log").write_text(log)
    bad = re.findall(r"(?im)^.*(?:error:|simulation\(s\) aborted|no convergence).*$", log)
    if result.returncode or bad or not output.exists():
        raise RuntimeError(f"SPICE failed in {directory}: exit={result.returncode}, {bad[:3]}")
    values = np.loadtxt(output, ndmin=2)
    if values.shape != (len(commands), len(expressions) + 1):
        raise RuntimeError(f"SPICE output shape {values.shape} in {directory}")
    if not np.array_equal(values[:, 0], np.arange(len(commands))) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid SPICE rows in {directory}")
    warnings = sorted(set(re.findall(r"(?im)^.*warning:.*$", log)))
    meta = dict(rows=len(commands), elapsed_s=elapsed, failures=0, retries=0,
                warnings=warnings, returncode=result.returncode,
                dynamic_gmin_attempts=log.count("Starting dynamic gmin stepping"),
                dynamic_gmin_completions=log.count("Dynamic gmin stepping completed"),
                source_stepping_attempts=log.count("Starting source stepping"))
    save_json(directory / "execution.json", meta)
    return values[:, 1:], meta


def single_batch(ctx, case, rows, tag):
    directory = ctx.scratch / tag
    deck = single_deck(ctx, case, directory)
    branch, meta = run_batch(ctx, tag, deck, [single_commands(case, row) for row in rows],
                             [f"i(v{t})" for t in "dgsb"])
    # Bias sources point from device terminal to ground; negate their branch currents.
    return -branch, meta


def smoke(ctx):
    all_cases = {}
    for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
        row = nominal_row(ctx, case)
        deck = single_deck(ctx, case, ctx.netlists)
        example = deck + "\n.control\nset numdgt=17\nop\nprint i(vd) i(vg) i(vs) i(vb)\nquit 0\n.endc\n.end\n"
        (ctx.netlists / f"{case}.cir").write_text(example)
        changed = dict(row, vth0=row["vth0"] * 1.1)
        values, meta = single_batch(ctx, case, [row, changed, row], f"part1/{case}")
        np.testing.assert_allclose(values[0], values[2], rtol=1e-10, atol=1e-18)
        if np.isclose(values[0, 0], values[1, 0], rtol=1e-3, atol=1e-18):
            raise RuntimeError(f"vth0 change has no measurable effect in {case}")
        # Compare the batched nominal result against a fresh, unaltered model-card run.
        fresh, _ = run_batch(ctx, f"part1/{case}_fresh", deck.replace(
            include_path(ctx, ctx.netlists), include_path(ctx, ctx.scratch / f"part1/{case}_fresh")),
            [[]], [f"i(v{t})" for t in "dgsb"])
        np.testing.assert_allclose(values[0], -fresh[0], rtol=1e-7, atol=1e-16)
        check = kcl_check(values, ctx.config)
        assert check["scale_aware_failures"] == 0
        assert values[0, 0] < 0 if case.startswith("pmos") else values[0, 0] > 0
        all_cases[case] = dict(nominal_currents_A=values[0], changed_vth0_currents_A=values[1],
                               nominal_restored=True, kcl=check, execution=meta)
    ctx.check("part1_smoke", all_cases)
    return all_cases
