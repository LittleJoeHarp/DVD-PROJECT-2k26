"""Static CMOS topologies, shared corners, and complete-gate SPICE references."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import time

import numpy as np
import pandas as pd

from common import FEATURES, OPTIONS, PROCESS, TARGETS, kcl_check, sha256
from spice import include_path, number, process_commands, run_batch

GATES = ("INV", "NAND2", "NOR2", "AND2", "OR2")


def patterns(gate):
    return list(product((0, 1), repeat=1 if gate == "INV" else 2))


def logic(gate, bits):
    if gate == "INV":
        return 1 - bits[0]
    if gate in ("NAND2", "AND2"):
        value = int(all(bits))
    else:
        value = int(any(bits))
    return 1 - value if gate in ("NAND2", "NOR2") else value


def topology(gate):
    """Devices are (name, polarity, drain, gate, source, bulk), with flat node names."""
    if gate == "INV":
        return [("p1", "pmos", "out", "a", "vdd", "vdd"),
                ("n1", "nmos", "out", "a", "0", "0")]
    if gate in ("NAND2", "AND2"):
        devices = [("p1", "pmos", "out", "a", "vdd", "vdd"),
                   ("p2", "pmos", "out", "b", "vdd", "vdd"),
                   ("n1", "nmos", "out", "a", "x", "0"),
                   ("n2", "nmos", "x", "b", "0", "0")]
    elif gate in ("NOR2", "OR2"):
        devices = [("p1", "pmos", "y", "a", "vdd", "vdd"),
                   ("p2", "pmos", "out", "b", "y", "vdd"),
                   ("n1", "nmos", "out", "a", "0", "0"),
                   ("n2", "nmos", "out", "b", "0", "0")]
    else:
        raise ValueError(gate)
    if gate in ("AND2", "OR2"):
        devices = [(name, polarity, *("mid" if node == "out" else node for node in nodes))
                   for name, polarity, *nodes in devices]
        devices += [("p3", "pmos", "out", "mid", "vdd", "vdd"),
                    ("n3", "nmos", "out", "mid", "0", "0")]
    return devices


def width(ctx, polarity):
    return ctx.config[f"{polarity}_gate_width_m"]


def device_lines(ctx, devices, sense=False):
    lines = []
    for name, polarity, *nodes in devices:
        terminals = nodes
        if sense:
            terminals = [f"{name}_{t}" for t in "dgsb"]
            lines += [f"V_{name}_{t} {node} {private} 0" for t, node, private in zip("dgsb", nodes, terminals)]
        lines.append(f"M_{name} {' '.join(terminals)} {polarity} W={number(width(ctx, polarity))} "
                     f"L={number(ctx.config['length_m'])}")
    return lines


def gate_deck(ctx, gate, directory, sense=True):
    lines = [f"Complete static CMOS {gate}", f'.include "{include_path(ctx, directory)}"', OPTIONS,
             ".temp 27", "VDD vdd 0 1", "V_a a 0 0"]
    if gate != "INV":
        lines.append("V_b b 0 0")
    lines += device_lines(ctx, topology(gate), sense=sense)
    return "\n".join(lines)


def corners(ctx):
    path = ctx.data / "corners.csv"
    rng = np.random.default_rng(ctx.config["seed"] + 200)
    n = ctx.config["corners"]
    sampled = dict(corner_id=np.arange(n), vdd=rng.uniform(*ctx.config["vdd_range_V"], n),
                   temp=rng.uniform(*ctx.config["temperature_range_C"], n))
    for polarity in ("nmos", "pmos"):
        for parameter in PROCESS:
            sampled[f"{polarity}_{parameter}"] = ctx.nominal[polarity][parameter] * rng.uniform(.9, 1.1, n)
    expected = pd.DataFrame(sampled)
    if path.exists():
        stored = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_allclose(stored, expected, rtol=1e-15, atol=0)
        return stored
    expected.to_csv(path, index=False, float_format="%.17g")
    return expected


def corner_commands(corner):
    changes = [f'set temp = {number(corner["temp"])}', f'alter vdd = {number(corner["vdd"])}']
    for polarity in ("nmos", "pmos"):
        changes += process_commands(polarity, {name: corner[f"{polarity}_{name}"] for name in PROCESS})
    return changes


def feature_rows(ctx, gate, bits, corner, voltages):
    rows = []
    for name, polarity, *nodes in topology(gate):
        terminal_voltages = {f"v{t}": voltages[node] for t, node in zip("dgsb", nodes)}
        high = terminal_voltages["vg"] >= .5 * corner["vdd"]
        case = f"{polarity}_{'on' if high == (polarity == 'nmos') else 'off'}"
        rows.append(dict(corner_id=int(corner["corner_id"]), gate=gate, input_pattern="".join(map(str, bits)),
                         device=name, polarity=polarity, case=case,
                         vdd=corner["vdd"], temp=corner["temp"], w=width(ctx, polarity), **terminal_voltages,
                         **{name: corner[f"{polarity}_{name}"] for name in PROCESS},
                         **{f"node_{t}": node for t, node in zip("dgsb", nodes)},
                         gate_rail_deviation_V=abs(terminal_voltages["vg"] - (corner["vdd"] if high else 0)),
                         voltage_origin="isolated_gate_SPICE"))
    return rows


def simulate_gate_corner(ctx, gate, corner):
    tag = f"part4/{gate}/{int(corner['corner_id']):03d}"
    devices = topology(gate)
    node_names = sorted(set(node for device in devices for node in device[2:]) - {"0"})
    inputs = ["a"] if gate == "INV" else ["a", "b"]
    expressions = ["i(vdd)", *[f"v({node})" for node in node_names],
                   *[f"i(v_{node})" for node in inputs],
                   *[f"i(v_{device[0]}_{t})" for device in devices for t in "dgsb"]]
    changes = [[f"alter v_{node} = {number(bit * corner['vdd'])}" for node, bit in zip(inputs, bits)]
               for bits in patterns(gate)]
    values, meta = run_batch(ctx, tag, gate_deck(ctx, gate, ctx.scratch / tag), changes,
                             expressions, prefix=corner_commands(corner))
    references, terminals = [], []
    for bits, value in zip(patterns(gate), values):
        voltages = dict(zip(node_names, value[1:1 + len(node_names)]), **{"0": 0.0})
        offset = 1 + len(node_names)
        input_currents = dict(zip(inputs, -value[offset:offset+len(inputs)]))
        device_currents = value[offset+len(inputs):].reshape(-1, 4)
        kcl = kcl_check(device_currents, ctx.config)
        assert kcl["scale_aware_failures"] == 0, (gate, corner["corner_id"], bits, kcl)
        boundary = sum(device_currents[j, k] for j, device in enumerate(devices)
                       for k, node in enumerate(device[2:]) if node == "vdd")
        leakage = -value[0]
        assert abs(boundary-leakage) <= 1e-14 + 1e-6*abs(leakage)
        expected = logic(gate, bits)
        fraction = voltages["out"] / corner["vdd"]
        valid = fraction >= ctx.config["logic_high_fraction"] if expected else fraction <= ctx.config["logic_low_fraction"]
        if not valid:
            raise RuntimeError(f"Incorrect {gate} logic at corner {corner['corner_id']}, inputs {bits}: {fraction}")
        ref = dict(corner_id=int(corner["corner_id"]), gate=gate, input_pattern="".join(map(str, bits)),
                   vdd=corner["vdd"], temp=corner["temp"], leakage_spice_A=leakage,
                   boundary_sum_A=boundary, logic_expected=expected, logic_observed=int(fraction >= .5),
                   **{f"v_{node}": voltage for node, voltage in voltages.items() if node != "0"},
                   **{f"input_{node}_draw_A": current for node, current in input_currents.items()})
        references.append(ref)
        features = feature_rows(ctx, gate, bits, corner, voltages)
        for row, currents in zip(features, device_currents):
            row.update(dict(zip(TARGETS, currents)))
            terminals.append(row)
    return references, terminals, meta


def references(ctx):
    shared = corners(ctx)
    for gate in GATES:
        example = gate_deck(ctx, gate, ctx.netlists)
        nodes = sorted(set(node for d in topology(gate) for node in d[2:]) - {"0"})
        example += "\n.control\nset numdgt=17\nop\nprint i(vdd) " + " ".join(f"v({n})" for n in nodes)
        example += "\nquit 0\n.endc\n.end\n"
        (ctx.netlists / f"{gate}.cir").write_text(example)
    start = time.perf_counter()
    all_refs, all_devices, executions = [], [], []
    with ThreadPoolExecutor(max_workers=ctx.config["workers"]) as pool:
        jobs = [pool.submit(simulate_gate_corner, ctx, gate, corner)
                for corner in shared.to_dict("records") for gate in GATES]
        for complete, job in enumerate(as_completed(jobs), 1):
            refs, devices, meta = job.result()
            all_refs.extend(refs)
            all_devices.extend(devices)
            executions.append(meta)
            if complete == 1 or complete % 200 == 0:
                print(f"Gate references: {complete}/{len(jobs)} corner/gate batches, "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
    refs = pd.DataFrame(all_refs).sort_values(["gate", "input_pattern", "corner_id"])
    devices = pd.DataFrame(all_devices).sort_values(["gate", "input_pattern", "corner_id", "device"])
    refs.to_csv(ctx.data / "gate_references.csv", index=False, float_format="%.17g")
    devices.to_csv(ctx.data / "gate_device_references.csv", index=False, float_format="%.17g")
    assert len(refs) == 3600 and len(devices) == 16800
    assert (refs.groupby(["gate", "input_pattern"]).size() == 200).all()
    assert not refs.duplicated(["corner_id", "gate", "input_pattern"]).any()
    checks = dict(records=len(refs), device_records=len(devices), corners=len(shared),
                  cases=len(refs.groupby(["gate", "input_pattern"])),
                  truth_table_failures=int((refs.logic_expected != refs.logic_observed).sum()),
                  max_supply_boundary_difference_A=float(abs(refs.leakage_spice_A-refs.boundary_sum_A).max()),
                  terminal_kcl=kcl_check(devices.loc[:, TARGETS], ctx.config),
                  simulation_failures=sum(e["failures"] for e in executions),
                  retries=sum(e["retries"] for e in executions), elapsed_s=time.perf_counter()-start,
                  corner_sha256=sha256(ctx.data / "corners.csv"), reference_sha256=sha256(ctx.data / "gate_references.csv"),
                  complete_AND2_OR2=True, input_supplies_separate=True,
                  warnings=sorted(set(w for e in executions for w in e["warnings"])))
    ctx.check("part4_gates", checks)
    print("Part 4 passed: 3,600 references, 16,800 measured devices, all truth tables and supply boundaries.", flush=True)


def recheck(ctx):
    refs = pd.read_csv(ctx.data / "gate_references.csv", dtype={"input_pattern": str}, float_precision="round_trip")
    shared = corners(ctx)
    truth_tables = {"INV": [1, 0], "NAND2": [1, 1, 1, 0], "NOR2": [1, 0, 0, 0],
                    "AND2": [0, 0, 0, 1], "OR2": [0, 1, 1, 1]}
    for gate, truth in truth_tables.items():
        for bits, expected in zip(patterns(gate), truth):
            pattern = "".join(map(str, bits))
            rows = refs[(refs.gate == gate) & (refs.input_pattern == pattern)]
            assert len(rows) == 200 and set(rows.corner_id) == set(range(200))
            assert (rows.logic_observed == expected).all()
            if gate in ("NAND2", "AND2"):
                assert rows.v_x.notna().all()
            if gate in ("NOR2", "OR2"):
                assert rows.v_y.notna().all()
            if gate in ("AND2", "OR2"):
                assert rows.v_mid.notna().all()
    differences = []
    for index in sorted({0, int(shared.temp.idxmin()), int(shared.temp.idxmax())}):
        corner = shared.iloc[index].to_dict()
        for gate in GATES:
            inputs = ["a"] if gate == "INV" else ["a", "b"]
            changes = [[f"alter v_{node} = {number(bit*corner['vdd'])}" for node, bit in zip(inputs, bits)]
                       for bits in patterns(gate)]
            tag = f"part4/recheck/{gate}_{index}"
            values, _ = run_batch(ctx, tag, gate_deck(ctx, gate, ctx.scratch / tag, sense=False),
                                 changes, ["i(vdd)", "v(out)"], prefix=corner_commands(corner))
            measured = refs[(refs.gate == gate) & (refs.corner_id == index)].sort_values("input_pattern")
            np.testing.assert_allclose(-values[:, 0], measured.leakage_spice_A, rtol=1e-5, atol=1e-14)
            np.testing.assert_allclose(values[:, 1], measured.v_out, rtol=1e-5, atol=1e-7)
            differences.extend(abs(-values[:, 0] - measured.leakage_spice_A))
    ctx.check("part4_recheck", dict(explicit_truth_tables_passed=True, complete_coverage=True,
                                    internal_nodes_present=True, sensing_source_comparison_records=len(differences),
                                    max_uninstrumented_supply_difference_A=max(differences)))
    print("Part 4 recheck passed: explicit truth tables and sensor-free netlist comparisons.", flush=True)


def predict_devices(features, models):
    """Predict using feature columns only; no gate/reference current is accepted."""
    from threadpoolctl import threadpool_limits
    result = features.copy()
    predictions = np.empty((len(features), 4))
    with threadpool_limits(limits=2):
        for case, model in models.items():
            mask = (features.case == case).to_numpy()
            if mask.any():
                predictions[mask] = model.predict(features.loc[mask, list(FEATURES)])
    for j, target in enumerate(TARGETS):
        result[f"ml_{target}"] = predictions[:, j]
    boundary = np.zeros(len(features))
    for j, terminal in enumerate("dgsb"):
        boundary += predictions[:, j] * (features[f"node_{terminal}"] == "vdd").to_numpy()
    result["vdd_contribution_A"] = boundary
    for port in ("a", "b"):
        contribution = np.zeros(len(features))
        for j, terminal in enumerate("dgsb"):
            contribution += predictions[:, j] * (features[f"node_{terminal}"] == port).to_numpy()
        result[f"input_{port}_contribution_A"] = contribution
    return result


def predict_gate(ctx, gate, bits, corner, voltages, models):
    """Reusable isolated-gate prediction with explicitly supplied SPICE voltages."""
    rows = pd.DataFrame(feature_rows(ctx, gate, bits, corner, voltages))
    predicted = predict_devices(rows, models)
    return float(predicted.vdd_contribution_A.sum()), predicted


def predictions(ctx):
    import joblib
    import matplotlib.pyplot as plt
    from common import CASES
    from metrics import metrics, paired_columns, parity_plot

    hashes = {case: sha256(ctx.results / "trained" / f"{case}.joblib") for case in CASES}
    models = {case: joblib.load(ctx.results / "trained" / f"{case}.joblib") for case in CASES}
    refs = pd.read_csv(ctx.data / "gate_references.csv", dtype={"input_pattern": str}, float_precision="round_trip")
    shared = corners(ctx).set_index("corner_id", drop=False)
    # Strip all measured current columns before constructing inputs for inference.
    allowed = ["corner_id", "gate", "input_pattern"] + [c for c in refs.columns if c.startswith("v_")]
    features = []
    for ref in refs[allowed].to_dict("records"):
        gate = ref["gate"]
        nodes = set(n for device in topology(gate) for n in device[2:]) - {"0"}
        voltages = {node: ref[f"v_{node}"] for node in nodes}
        voltages["0"] = 0.0
        features.extend(feature_rows(ctx, gate, tuple(map(int, ref["input_pattern"])),
                                     shared.loc[ref["corner_id"]].to_dict(), voltages))
    device_predictions = predict_devices(pd.DataFrame(features), models)
    keys = ["gate", "input_pattern", "corner_id"]
    totals = device_predictions.groupby(keys, as_index=False)[
        ["vdd_contribution_A", "input_a_contribution_A", "input_b_contribution_A"]].sum().rename(
        columns={"vdd_contribution_A": "leakage_ml_A", "input_a_contribution_A": "input_a_ml_A",
                 "input_b_contribution_A": "input_b_ml_A"})
    paired = refs.merge(totals, on=keys, validate="one_to_one")
    paired["ml_voltage_origin"] = "isolated_gate_SPICE_record_reused"
    for node in ("out", "x", "y", "mid"):
        paired[f"ml_v_{node}_from_spice"] = paired[f"v_{node}"]
    threshold = ctx.config["near_zero_current_A"]
    for key, values in paired_columns(paired.leakage_spice_A, paired.leakage_ml_A, threshold).items():
        paired[key] = values
    summaries = [dict(gate=key[0], input_pattern=key[1],
                      **metrics(rows.leakage_spice_A, rows.leakage_ml_A, threshold))
                 for key, rows in paired.groupby(["gate", "input_pattern"])]
    table = pd.DataFrame(summaries)
    device_predictions.to_csv(ctx.results / "tables" / "gate_device_predictions.csv", index=False, float_format="%.17g")
    paired.to_csv(ctx.results / "tables" / "gate_predictions.csv", index=False, float_format="%.17g")
    table.to_csv(ctx.results / "tables" / "gate_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot = parity_plot(ax, paired.leakage_spice_A, paired.leakage_ml_A, "All 3,600 gate/input/corner records")
    fig.savefig(ctx.results / "plots" / "gate_parity.png", dpi=170)
    plt.close(fig)
    overall = metrics(paired.leakage_spice_A, paired.leakage_ml_A, threshold)
    ranked = table.sort_values("mean_error_pct")
    largest = paired.nlargest(20, "percentage_error").merge(
        shared.drop(columns=["vdd", "temp"]).reset_index(drop=True), on="corner_id", validate="many_to_one")
    largest.to_csv(ctx.results / "tables" / "gate_largest_errors.csv", index=False)
    support = {}
    from data import load_training_data
    for case in CASES:
        training, labels, _ = load_training_data(ctx, case)
        training = training[labels == "train"]
        support[case] = dict(training_rows=len(training),
            both_drain_source_within_10mV_of_vdd=int(((training.vdd-training.vd < .01) & (training.vdd-training.vs < .01)).sum()),
            both_within_10mV_of_ground=int(((training.vd < .01) & (training.vs < .01)).sum()))
    worst = paired.loc[paired.percentage_error.idxmax()]
    ctx.check("gate_error_analysis", dict(best_case=ranked.iloc[0].to_dict(), worst_case=ranked.iloc[-1].to_dict(),
        largest_errors_csv="results/tables/gate_largest_errors.csv",
        predicted_current_kcl=kcl_check(device_predictions[[f"ml_{t}" for t in TARGETS]], ctx.config),
        rail_routing_max_deviation_V=float(device_predictions.gate_rail_deviation_V.max()),
        rail_equal_training_support=support,
        worst_record=worst[["gate", "input_pattern", "corner_id", "vdd", "temp", "leakage_spice_A", "leakage_ml_A", "percentage_error"]].to_dict(),
        interpretation="Worst evaluated corners are retained for diagnosis. Joint current coordinates enforce KCL; boundary coverage includes the supplementary transistor-only population.",
        note="Joint terminal-coordinate reconstruction enforces KCL; no gate-target training or calibration."))
    assert len(paired) == 3600 and len(table) == 18
    assert hashes == {case: sha256(ctx.results / "trained" / f"{case}.joblib") for case in CASES}
    ctx.check("part5_predictions", dict(overall=overall, plot=plot, rows=len(paired), table_rows=len(table),
                                       models_unchanged=True, model_hashes=hashes,
                                       input_columns=list(FEATURES), node_voltage_origin="SPICE, not ML-predicted"))
    print(f"Part 5: {len(paired)} predictions, mean accuracy={overall['mean_accuracy_pct']:.2f}%, "
          f"median error={overall['median_error_pct']:.2f}%", flush=True)
    recheck_predictions(ctx, models)


def recheck_predictions(ctx, models=None):
    import joblib
    from common import CASES
    from metrics import metrics
    models = models or {case: joblib.load(ctx.results / "trained" / f"{case}.joblib") for case in CASES}
    paired = pd.read_csv(ctx.results / "tables" / "gate_predictions.csv", dtype={"input_pattern": str}, float_precision="round_trip")
    devices = pd.read_csv(ctx.results / "tables" / "gate_device_predictions.csv", dtype={"input_pattern": str}, float_precision="round_trip")
    shared = corners(ctx).set_index("corner_id", drop=False)
    checked = []
    for gate in GATES:
        for pattern in ["".join(map(str, bits)) for bits in patterns(gate)]:
            rows = paired[(paired.gate == gate) & (paired.input_pattern == pattern)]
            row = rows[rows.corner_id == 0].iloc[0]
            nodes = set(n for device in topology(gate) for n in device[2:]) - {"0"}
            voltages = {n: row[f"v_{n}"] for n in nodes}
            voltages["0"] = 0.0
            value, detail = predict_gate(ctx, gate, tuple(map(int, pattern)), shared.loc[0].to_dict(), voltages, models)
            np.testing.assert_allclose(value, row.leakage_ml_A, rtol=1e-9, atol=1e-18)
            checked.append(dict(gate=gate, input_pattern=pattern, leakage_ml_A=value,
                                boundary_terms=[f"{d[0]}.{t}" for d in topology(gate)
                                                for t, n in zip("dgsb", d[2:]) if n == "vdd"]))
    independent_sum = np.zeros(len(devices))
    for terminal in "dgsb":
        independent_sum += devices[f"ml_i{terminal}"] * (devices[f"node_{terminal}"] == "vdd")
    np.testing.assert_allclose(independent_sum, devices.vdd_contribution_A, rtol=1e-14, atol=1e-25)
    for node in ("out", "x", "y", "mid"):
        np.testing.assert_allclose(paired[f"ml_v_{node}_from_spice"], paired[f"v_{node}"], equal_nan=True)
    ctx.check("part5_recheck", dict(representative_cases=checked, signed_boundary_sum_verified=True,
                                    voltage_origin_labels_verified=True,
                                    recomputed=metrics(paired.leakage_spice_A, paired.leakage_ml_A,
                                                       ctx.config["near_zero_current_A"])))
    print("Part 5 recheck passed: all 18 mappings, signed supply contributions, and SPICE voltage labels.", flush=True)
