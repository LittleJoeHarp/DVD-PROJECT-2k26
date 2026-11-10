"""Independent PVT sampling, transistor datasets, fixed splits, and diagnostics."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CASES, FEATURES, PROCESS, TARGETS, kcl_check, nominal_row, sha256
from spice import number, run_batch, single_batch, single_deck


def sample_case(ctx, case, count, seed):
    rng = np.random.default_rng(seed)
    cfg = ctx.config
    vdd = rng.uniform(*cfg["vdd_range_V"], count)
    nominal = ctx.nominal[case.split("_")[0]]
    pmos, on = case.startswith("pmos"), case.endswith("_on")
    values = dict(vdd=vdd, temp=rng.uniform(*cfg["temperature_range_C"], count),
                  w=rng.uniform(*cfg["width_range_m"], count),
                  vd=rng.uniform(0, vdd), vs=rng.uniform(0, vdd),
                  vg=vdd if (pmos and not on) or (not pmos and on) else np.zeros(count),
                  vb=vdd if pmos else np.zeros(count))
    values.update({key: nominal[key] * rng.uniform(*cfg["process_factor_range"], count)
                   for key in PROCESS})
    return pd.DataFrame(values, columns=FEATURES)


def strata(frame):
    """Input-only coarse bias regime and temperature band; never current labels."""
    low = np.minimum(frame.vd, frame.vs) / frame.vdd
    high = np.maximum(frame.vd, frame.vs) / frame.vdd
    regime = np.select([high-low < .03, low < .05, high > .95],
                       ["near_equal", "lower_rail", "upper_rail"], default="interior")
    band = np.digitize(frame.temp, [15, 70])
    return np.array([f"{r}_T{b}" for r, b in zip(regime, band)])


def make_split(frame, seed):
    labels = strata(frame)
    ids = np.arange(len(frame))
    training, holdout = train_test_split(ids, test_size=.30, random_state=seed, stratify=labels)
    validation, test = train_test_split(holdout, test_size=.50, random_state=seed+1,
                                       stratify=labels[holdout])
    split = np.empty(len(frame), dtype=object)
    split[training], split[validation], split[test] = "train", "validation", "test"
    return pd.DataFrame(dict(row_id=ids, split=split, stratum=labels))


def sample_boundary(ctx, case, count, seed):
    """Supplementary single-device design, independent of gate/circuit references."""
    frame = sample_case(ctx, case, count, seed)
    rng = np.random.default_rng(seed+100)
    vdd = frame.vdd.to_numpy()
    regime = rng.choice(5, count, p=[.4, .1, .2, .2, .1])
    rail = rng.integers(0, 2, count)
    def offset():
        value = np.power(10.0, rng.uniform(-8, np.log10(.03), count))
        value[rng.random(count) < .25] = 0
        return value
    d, s = frame.vd.to_numpy().copy(), frame.vs.to_numpy().copy()
    a, b = offset(), offset()
    mask = regime == 0  # Both terminals close to the same rail, including exact equality.
    d[mask] = np.where(rail[mask] == 0, a[mask], 1-a[mask]) * vdd[mask]
    s[mask] = np.where(rail[mask] == 0, b[mask], 1-b[mask]) * vdd[mask]
    mask = regime == 1  # Small Vds at interior common-mode voltages.
    center = rng.uniform(0, 1, count)
    d[mask] = center[mask] * vdd[mask]
    s[mask] = np.clip(center[mask] + rng.choice([-1, 1], mask.sum()) * a[mask], 0, 1) * vdd[mask]
    mask = regime == 2  # Opposite rails, important for OFF-state supply currents.
    d[mask], s[mask] = a[mask]*vdd[mask], (1-b[mask])*vdd[mask]
    mask = regime == 3  # One rail and one independent interior voltage (stack support).
    d[mask] = np.where(rail[mask] == 0, a[mask], 1-a[mask]) * vdd[mask]
    swap = rng.random(count) < .5
    d[swap], s[swap] = s[swap].copy(), d[swap].copy()
    frame["vd"], frame["vs"] = d, s
    # Half use the actual gate-library geometry; half retain continuous widths.
    fixed = rng.random(count) < .5
    frame.loc[fixed, "w"] = ctx.config[f"{case.split('_')[0]}_gate_width_m"]
    return frame


def load_training_data(ctx, case):
    uniform = pd.read_csv(ctx.data / f"{case}.csv", float_precision="round_trip")
    usplit = pd.read_csv(ctx.data / f"{case}_splits.csv")
    supplemental = pd.read_csv(ctx.data / f"{case}_boundary.csv", float_precision="round_trip")
    ssplit = pd.read_csv(ctx.data / f"{case}_boundary_splits.csv")
    frame = pd.concat([uniform, supplemental], ignore_index=True)
    labels = np.concatenate([usplit.split, ssplit.split])
    origins = np.array(["uniform"]*len(uniform) + ["boundary"]*len(supplemental))
    return frame, labels, origins


def validate_dataset(ctx, case, frame, expected_count=None):
    cfg = ctx.config
    assert list(frame.columns) == list(FEATURES + TARGETS)
    assert len(frame) == (cfg["samples_per_case"] if expected_count is None else expected_count)
    assert np.isfinite(frame.to_numpy()).all()
    assert not frame.duplicated().any()
    assert not frame.loc[:, FEATURES].duplicated().any()
    for column, limits in [("vdd", cfg["vdd_range_V"]), ("temp", cfg["temperature_range_C"]),
                           ("w", cfg["width_range_m"])]:
        assert frame[column].between(*limits).all(), column
    for column in ("vd", "vs"):
        assert ((frame[column] >= 0) & (frame[column] <= frame.vdd)).all()
    pmos, on = case.startswith("pmos"), case.endswith("_on")
    np.testing.assert_array_equal(frame.vg, frame.vdd if pmos != on else np.zeros(len(frame)))
    np.testing.assert_array_equal(frame.vb, frame.vdd if pmos else np.zeros(len(frame)))
    nominal = ctx.nominal[case.split("_")[0]]
    for parameter in PROCESS:
        assert (frame[parameter] / nominal[parameter]).between(*cfg["process_factor_range"]).all()
    kcl = kcl_check(frame.loc[:, TARGETS], cfg)
    assert kcl["scale_aware_failures"] == 0, (case, kcl)
    return dict(rows=len(frame), columns=list(frame.columns), finite=True,
                duplicate_rows=0, bounds_passed=True,
                vd_less_than_vs=int((frame.vd < frame.vs).sum()), kcl=kcl)


def generate(ctx):
    start = time.perf_counter()
    count = ctx.config["samples_per_case"]
    size = ctx.config["batch_size"]
    frames = {case: sample_case(ctx, case, count, ctx.config["seed"] + i)
              for i, case in enumerate(CASES)}
    outputs = {case: np.empty((count, 4)) for case in CASES}
    batches = {case: [] for case in CASES}
    jobs = {}
    with ThreadPoolExecutor(max_workers=ctx.config["workers"]) as pool:
        for case in CASES:
            for first in range(0, count, size):
                rows = frames[case].iloc[first:first + size].to_dict("records")
                job = pool.submit(single_batch, ctx, case, rows, f"part2/{case}/{first:05d}")
                jobs[job] = (case, first)
        for completed, job in enumerate(as_completed(jobs), 1):
            case, first = jobs[job]
            values, meta = job.result()
            outputs[case][first:first + len(values)] = values
            batches[case].append(meta)
            if completed == 1 or completed % 20 == 0 or completed == len(jobs):
                elapsed = time.perf_counter() - start
                print(f"Transistors: {completed}/{len(jobs)} batches, {elapsed:.1f}s", flush=True)
    checks = {}
    for i, case in enumerate(CASES):
        frame = frames[case]
        frame[list(TARGETS)] = outputs[case]
        path = ctx.data / f"{case}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        checks[case] = validate_dataset(ctx, case, frame)
        split = make_split(frame, ctx.config["seed"]+100+i)
        labels = split.split.to_numpy()
        split.to_csv(ctx.data / f"{case}_splits.csv", index=False)
        checks[case].update(dataset_sha256=sha256(path),
                            split_counts={key: int((labels == key).sum()) for key in set(labels)},
                            simulation_failures=sum(b["failures"] for b in batches[case]),
                            retries=sum(b["retries"] for b in batches[case]),
                            seed=ctx.config["seed"] + i)
    ctx.check("part2_datasets", checks)
    supplementary(ctx)
    trends(ctx)
    print(f"Part 2 passed: four {count}-row datasets, splits, KCL and trend plots; "
          f"{time.perf_counter()-start:.1f}s", flush=True)


def supplementary(ctx):
    count = ctx.config["supplementary_samples_per_case"]
    size = ctx.config["batch_size"]
    checks = {}
    for index, case in enumerate(CASES):
        frame = sample_boundary(ctx, case, count, ctx.config["seed"]+1000+index)
        output = np.empty((count, 4))
        execution = []
        with ThreadPoolExecutor(max_workers=ctx.config["workers"]) as pool:
            jobs = {pool.submit(single_batch, ctx, case, frame.iloc[first:first+size].to_dict("records"),
                                f"boundary/{case}/{first:05d}"): first for first in range(0, count, size)}
            for job in as_completed(jobs):
                first = jobs[job]
                values, meta = job.result()
                output[first:first+len(values)] = values
                execution.append(meta)
        frame[list(TARGETS)] = output
        validate_dataset(ctx, case, frame, expected_count=count)
        assert np.isfinite(frame.to_numpy()).all() and not frame.duplicated().any()
        assert ((frame.vd >= 0) & (frame.vd <= frame.vdd) & (frame.vs >= 0) & (frame.vs <= frame.vdd)).all()
        kcl = kcl_check(output, ctx.config)
        assert kcl["scale_aware_failures"] == 0
        path = ctx.data / f"{case}_boundary.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        split = make_split(frame, ctx.config["seed"]+1100+index)
        split.to_csv(ctx.data / f"{case}_boundary_splits.csv", index=False)
        counts = pd.crosstab(split.stratum, split.split)
        assert (counts > 0).all().all()
        checks[case] = dict(rows=count, sha256=sha256(path), kcl=kcl,
                            split_counts=split.split.value_counts().to_dict(),
                            strata=counts.to_dict(), gate_jitter_rows=int((frame.vg != (
                                frame.vdd if case in ("nmos_on", "pmos_off") else 0)).sum()),
                            failures=sum(e["failures"] for e in execution), retries=sum(e["retries"] for e in execution))
        print(f"Supplementary {case}: {count} rows, stratified splits, KCL passed", flush=True)
    ctx.check("boundary_datasets", checks)


def trends(ctx):
    """Controlled nominal-bias sweeps, kept separate from randomized training data."""
    rows_out = []
    for variable in ("temp", "vth0", "toxe"):
        fig, ax = plt.subplots(figsize=(6.6, 4.6), layout="constrained")
        for polarity in ("nmos", "pmos"):
            case = f"{polarity}_{'on' if variable == 'toxe' else 'off'}"
            row = nominal_row(ctx, case)
            points = np.linspace(-40, 125, 41) if variable == "temp" else row[variable] * np.linspace(.9, 1.1, 41)
            records = [dict(row, **{variable: float(value)}) for value in points]
            currents, _ = single_batch(ctx, case, records, f"part2/trends/{variable}_{polarity}")
            target = 1 if variable == "toxe" else 0
            x = points if variable == "temp" else points / row[variable]
            ax.semilogy(x, np.abs(currents[:, target]), label=polarity.upper())
            for record, value in zip(records, currents):
                rows_out.append(dict(sweep=variable, case=case, **record, **dict(zip(TARGETS, value))))
        ax.set(xlabel="Temperature (°C)" if variable == "temp" else f"{variable} / nominal {variable}",
               ylabel="|Ig| (A)" if variable == "toxe" else "|Id, OFF| (A)",
               title=f"Controlled {variable} sweep: VDD=1 V, W=64 nm, L=32 nm")
        ax.grid(True, which="both", alpha=.25)
        ax.legend()
        fig.savefig(ctx.results / "plots" / f"trend_{variable}.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(rows_out).to_csv(ctx.data / "diagnostic_sweeps.csv", index=False, float_format="%.17g")


def recheck(ctx):
    """Re-read deliverables and cross-check random rows using literal model cards."""
    checks = {}
    for case in CASES:
        frame = pd.read_csv(ctx.data / f"{case}.csv", float_precision="round_trip")
        check = validate_dataset(ctx, case, frame)
        split = pd.read_csv(ctx.data / f"{case}_splits.csv")
        assert np.array_equal(split.row_id, np.arange(len(frame)))
        assert split.split.value_counts().to_dict() == {"train": 7000, "validation": 1500, "test": 1500}
        maximum_difference = 0.0
        for index in (0, 4321, 9999):
            row = frame.iloc[index].to_dict()
            polarity = case.split("_")[0]
            tag = f"part2/recheck/{case}_{index}"
            directory = ctx.scratch / tag
            directory.mkdir(parents=True, exist_ok=True)
            original = (ctx.models / ctx.config["model_file"]).read_text()
            def replace_block(match):
                block = match.group(0)
                for name in PROCESS:
                    value = row[name] / (1e6 if name == "ndep" else 1)
                    block, changed = re.subn(r"(?i)(\b" + name + r"\s*=\s*)[-+0-9.eE]+",
                                            lambda m: m.group(1) + number(value), block)
                    assert changed == 1
                return block
            literal, changed = re.subn(r"(?ims)^\.model\s+" + polarity + r"\b.*?(?=^\.model|\Z)",
                                      replace_block, original)
            assert changed == 1
            (directory / "literal.pm").write_text(literal)
            deck = single_deck(ctx, case, directory, row)
            deck = re.sub(r'(?m)^\.include.*$', '.include "literal.pm"', deck)
            branch, _ = run_batch(ctx, tag, deck, [[]], [f"i(v{t})" for t in "dgsb"])
            expected = frame.loc[index, list(TARGETS)].to_numpy(dtype=float)
            # Model edits and altermod can differ by a few machine-epsilon
            # cancellation terms in branch currents; use the specified reltol.
            np.testing.assert_allclose(-branch[0], expected, rtol=1e-6, atol=1e-15)
            maximum_difference = max(maximum_difference, float(np.abs(-branch[0] - expected).max()))
        check["literal_model_card_comparison"] = {"rows": [0, 4321, 9999],
                                                  "maximum_difference_A": maximum_difference}
        checks[case] = check
    ctx.check("part2_recheck", checks)
    print("Part 2 recheck passed: saved CSVs/splits and 12 independent literal-card simulations.", flush=True)
