"""Balanced seven-XOR parity tree, expanded into 28 standard CMOS NAND2 gates."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OPTIONS, sha256, save_json
from gates import corner_commands, corners, device_lines, topology
from metrics import metrics, paired_columns, parity_plot
from spice import include_path, number, run_batch


def instances():
    gates = []
    previous = [f"d{i}" for i in range(8)]
    for level in range(3):
        next_level = []
        for index in range(len(previous)//2):
            a, b = previous[2*index:2*index+2]
            name = f"xor{level}_{index}"
            n1, n2, n3 = (f"{name}_n{i}" for i in (1, 2, 3))
            out = "p" if level == 2 else f"{name}_out"
            gates += [dict(name=f"{name}_g1", gate="NAND2", a=a, b=b, out=n1),
                      dict(name=f"{name}_g2", gate="NAND2", a=a, b=n1, out=n2),
                      dict(name=f"{name}_g3", gate="NAND2", a=b, b=n1, out=n3),
                      dict(name=f"{name}_g4", gate="NAND2", a=n2, b=n3, out=out)]
            next_level.append(out)
        previous = next_level
    assert len(gates) == 28 and previous == ["p"]
    return gates


def expanded_devices():
    devices = []
    for gate in instances():
        mapping = {"a": gate["a"], "b": gate["b"], "out": gate["out"],
                   "x": gate["name"] + "_x", "vdd": "vdd", "0": "0"}
        devices += [(gate["name"] + "_" + name, polarity, *[mapping[n] for n in nodes])
                    for name, polarity, *nodes in topology("NAND2")]
    assert len(devices) == 112
    return devices


def boolean_trace():
    gates = instances()
    traces, counts, expected_voltages = [], np.zeros((256, 4), dtype=int), []
    for pattern in range(256):
        values = {f"d{i}": (pattern >> i) & 1 for i in range(8)}
        outputs = []
        for gate in gates:
            a, b = values[gate["a"]], values[gate["b"]]
            out = 1 - a*b
            values[gate["out"]] = out
            counts[pattern, 2*a+b] += 1
            outputs.append(out)
            traces.append(dict(input_id=pattern, input_pattern=f"{pattern:08b}", instance=gate["name"],
                               gate="NAND2", gate_input_pattern=f"{a}{b}", output=out))
        assert values["p"] == sum((pattern >> i) & 1 for i in range(8)) % 2
        assert counts[pattern].sum() == 28
        expected_voltages.append(outputs)
    return pd.DataFrame(traces), counts, np.asarray(expected_voltages)


def internal_input_counts(trace):
    """Multiplicity of internal HIGH gate inputs, excluding external primary sources."""
    gates = {gate["name"]: gate for gate in instances()}
    primary = {f"d{i}" for i in range(8)}
    counts = np.zeros((256, 4, 2), dtype=int)
    for row in trace.itertuples():
        gate = gates[row.instance]
        for j, port in enumerate(("a", "b")):
            if gate[port] not in primary and row.gate_input_pattern[j] == "1":
                counts[row.input_id, int(row.gate_input_pattern, 2), j] += 1
    return counts


def circuit_deck(ctx, directory, initial_guesses=False):
    lines = ["Complete 8-bit odd-parity checker: seven XORs, 28 NAND2s, 112 MOSFETs",
             f'.include "{include_path(ctx, directory)}"', OPTIONS, ".temp 27", "VDD vdd 0 1"]
    lines += [f"V_d{i} d{i} 0 0" for i in range(8)]
    lines += device_lines(ctx, expanded_devices())
    if initial_guesses:
        lines.append(".param " + " ".join(f"guess{i}=0" for i in range(28)))
        # NODESET only initializes Newton iteration; NGSpice releases every hint
        # before solving the actual connected MOS circuit. No behavioral logic.
        lines.append(".nodeset " + " ".join(
            f"v({gate['out']})={{guess{i}}}" for i, gate in enumerate(instances())))
    return "\n".join(lines)


def simulate_corner(ctx, corner, expected, initial_guesses=True):
    gate_list = instances()
    output_nodes = [gate["out"] for gate in gate_list]
    stack_nodes = [gate["name"] + "_x" for gate in gate_list]
    folder = "part6_initial_guesses" if initial_guesses else "part6"
    tag = f"{folder}/{int(corner['corner_id']):03d}"
    changes = []
    for pattern in range(256):
        row = []
        if initial_guesses:
            row += [f"alterparam guess{i} = {number(bit*corner['vdd'])}" for i, bit in enumerate(expected[pattern])]
            row += ["reset", *corner_commands(corner)]
        row += [f"alter v_d{i} = {number(((pattern >> i) & 1) * corner['vdd'])}" for i in range(8)]
        changes.append(row)
    expressions = ["i(vdd)", *[f"v({n})" for n in output_nodes + stack_nodes]]
    values, meta = run_batch(ctx, tag, circuit_deck(ctx, ctx.scratch / tag, initial_guesses), changes,
                             expressions, prefix=[] if initial_guesses else corner_commands(corner))
    fractions = values[:, 1:29] / corner["vdd"]
    valid = np.where(expected == 1, fractions >= ctx.config["logic_high_fraction"],
                     fractions <= ctx.config["logic_low_fraction"])
    if not valid.all():
        bad = np.argwhere(~valid)[0]
        raise RuntimeError(f"Circuit logic failure at corner {corner['corner_id']}, input/gate {bad.tolist()}")
    rows = pd.DataFrame({"corner_id": int(corner["corner_id"]), "input_id": np.arange(256),
                         "input_pattern": [f"{p:08b}" for p in range(256)],
                         "vdd": corner["vdd"], "temp": corner["temp"],
                         "leakage_spice_A": -values[:, 0], "parity_expected": expected[:, -1],
                         "parity_observed": (fractions[:, -1] >= .5).astype(int)})
    for index, node in enumerate(output_nodes + stack_nodes, 1):
        rows[f"v_{node}"] = values[:, index]
    meta["logic_checks"] = int(valid.size)
    meta["logic_failures"] = 0
    meta["initial_voltage_hints"] = initial_guesses
    return rows, meta


def evaluate(ctx):
    start = time.perf_counter()
    corner_hash = sha256(ctx.data / "corners.csv")
    shared = corners(ctx)
    trace, counts, expected = boolean_trace()
    internal_counts = internal_input_counts(trace)
    trace.to_csv(ctx.data / "circuit_boolean_trace.csv", index=False)
    count_frame = pd.DataFrame(counts, columns=["NAND2_00", "NAND2_01", "NAND2_10", "NAND2_11"])
    count_frame.insert(0, "input_id", np.arange(256))
    count_frame.to_csv(ctx.data / "circuit_gate_counts.csv", index=False)
    pd.DataFrame(internal_counts.reshape(256, 8), columns=[f"{pattern}_{port}" for pattern in
        ("00", "01", "10", "11") for port in ("a", "b")]).to_csv(
            ctx.data / "circuit_internal_input_counts.csv", index_label="input_id")
    save_json(ctx.netlists / "parity8_instances.json", instances())
    example = circuit_deck(ctx, ctx.netlists) + "\n.control\nset numdgt=17\nop\nprint i(vdd) v(p)\nquit 0\n.endc\n.end\n"
    (ctx.netlists / "parity8.cir").write_text(example)

    # Full 256-pattern pilot provides an actual complete-circuit runtime estimate.
    first_rows, first_meta = simulate_corner(ctx, shared.iloc[0].to_dict(), expected)
    estimate = first_meta["elapsed_s"] * len(shared) / ctx.config["workers"]
    print(f"Circuit pilot: 256/256 patterns passed, {first_meta['elapsed_s']:.2f}s; "
          f"estimated simulator wall time with {ctx.config['workers']} workers: {estimate:.1f}s", flush=True)
    all_rows, executions = [first_rows], [first_meta]
    with ThreadPoolExecutor(max_workers=ctx.config["workers"]) as pool:
        jobs = [pool.submit(simulate_corner, ctx, corner, expected) for corner in shared.iloc[1:].to_dict("records")]
        for completed, job in enumerate(as_completed(jobs), 2):
            rows, meta = job.result()
            all_rows.append(rows)
            executions.append(meta)
            if completed % 20 == 0:
                print(f"Circuit: {completed}/200 corners ({completed*256:,} operating points), "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
    refs = pd.concat(all_rows, ignore_index=True).sort_values(["corner_id", "input_id"])
    refs.to_csv(ctx.data / "circuit_references.csv", index=False, float_format="%.17g")
    paired = refs[["corner_id", "input_id", "input_pattern", "vdd", "temp", "leakage_spice_A",
                   "parity_expected", "parity_observed", "v_p"]].copy()
    gate_results = pd.read_csv(ctx.results / "tables" / "gate_predictions.csv",
                               dtype={"input_pattern": str}, float_precision="round_trip")
    lookups = gate_results[gate_results.gate == "NAND2"].set_index(["corner_id", "input_pattern"])
    paired["leakage_ml_A"] = 0.0
    paired["isolated_spice_gate_sum_A"] = 0.0
    paired["internal_high_input_ml_A"] = 0.0
    paired["internal_high_input_spice_A"] = 0.0
    for corner_id, indices in paired.groupby("corner_id").groups.items():
        ml_lookup = np.array([lookups.loc[(corner_id, bits), "leakage_ml_A"] for bits in ("00", "01", "10", "11")])
        reference_lookup = np.array([lookups.loc[(corner_id, bits), "leakage_spice_A"] for bits in ("00", "01", "10", "11")])
        patterns_at_corner = paired.loc[indices, "input_id"].to_numpy()
        paired.loc[indices, "leakage_ml_A"] = counts[patterns_at_corner] @ ml_lookup
        # Separate diagnostic of the ideal-input approximation; never used to calculate ML.
        paired.loc[indices, "isolated_spice_gate_sum_A"] = counts[patterns_at_corner] @ reference_lookup
        ml_input = lookups.loc[[(corner_id, bits) for bits in ("00", "01", "10", "11")],
                               ["input_a_ml_A", "input_b_ml_A"]].to_numpy()
        spice_input = lookups.loc[[(corner_id, bits) for bits in ("00", "01", "10", "11")],
                                  ["input_a_draw_A", "input_b_draw_A"]].to_numpy()
        paired.loc[indices, "internal_high_input_ml_A"] = np.sum(
            internal_counts[patterns_at_corner] * ml_input[None, :, :], axis=(1, 2))
        paired.loc[indices, "internal_high_input_spice_A"] = np.sum(
            internal_counts[patterns_at_corner] * spice_input[None, :, :], axis=(1, 2))
    paired["uncorrected_leakage_ml_A"] = paired.leakage_ml_A
    paired["leakage_ml_A"] += paired.internal_high_input_ml_A
    paired["accounting_corrected_spice_A"] = paired.isolated_spice_gate_sum_A + paired.internal_high_input_spice_A
    paired["ml_voltage_origin"] = "isolated_gate_SPICE; Boolean interconnect; loading ignored"
    threshold = ctx.config["near_zero_current_A"]
    for key, values in paired_columns(paired.leakage_spice_A, paired.leakage_ml_A, threshold).items():
        paired[key] = values
    paired.to_csv(ctx.results / "tables" / "circuit_predictions.csv", index=False, float_format="%.17g")
    overall = metrics(paired.leakage_spice_A, paired.leakage_ml_A, threshold)
    per_pattern = [dict(input_id=int(key), input_pattern=f"{key:08b}",
                        **metrics(rows.leakage_spice_A, rows.leakage_ml_A, threshold))
                   for key, rows in paired.groupby("input_id")]
    pd.DataFrame(per_pattern).to_csv(ctx.results / "tables" / "circuit_by_input.csv", index=False)
    pd.DataFrame([dict(circuit="parity8", **overall)]).to_csv(ctx.results / "tables" / "circuit_results.csv", index=False)
    gate_overall = metrics(gate_results.leakage_spice_A, gate_results.leakage_ml_A, threshold)
    approximation = metrics(paired.leakage_spice_A, paired.isolated_spice_gate_sum_A, threshold)
    corrected_approximation = metrics(paired.leakage_spice_A, paired.accounting_corrected_spice_A, threshold)
    pd.DataFrame([dict(level="gates", **gate_overall), dict(level="circuit", **overall),
                  dict(level="ideal_input_approximation_with_SPICE_gate_currents", **approximation),
                  dict(level="accounting_corrected_SPICE_gate_currents", **corrected_approximation)]).to_csv(
                      ctx.results / "tables" / "gate_circuit_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot = parity_plot(ax, paired.leakage_spice_A, paired.leakage_ml_A, "8-bit parity: all 51,200 input/corner records")
    fig.savefig(ctx.results / "plots" / "circuit_parity.png", dpi=170)
    plt.close(fig)
    paired.nlargest(20, "percentage_error").to_csv(ctx.results / "tables" / "circuit_largest_errors.csv", index=False)
    assert len(refs) == len(paired) == 51200 and not paired.duplicated(["corner_id", "input_id"]).any()
    assert sha256(ctx.data / "corners.csv") == corner_hash
    ctx.check("part6_circuit", dict(records=len(paired), inputs=256, corners=200, gates=28, transistors=112,
        logic_checks=sum(e["logic_checks"] for e in executions), logic_failures=0,
        simulation_failures=sum(e["failures"] for e in executions), retries=sum(e["retries"] for e in executions),
        elapsed_s=time.perf_counter()-start, pilot_estimate_s=estimate, corner_sha256=corner_hash,
        shared_corners_unchanged=True, complete_circuit_simulated=True, overall=overall,
        ideal_input_approximation_diagnostic=approximation, plot=plot,
        accounting_corrected_spice_diagnostic=corrected_approximation,
        current_boundary="sum isolated VDD currents plus each internal HIGH input current once; primary input supplies excluded",
        warnings=sorted(set(w for e in executions for w in e["warnings"]))))
    print(f"Part 6 complete: 51,200 references and predictions, mean accuracy={overall['mean_accuracy_pct']:.2f}%", flush=True)
    recheck(ctx)


def recheck(ctx):
    paired = pd.read_csv(ctx.results / "tables" / "circuit_predictions.csv", dtype={"input_pattern": str}, float_precision="round_trip")
    trace, counts, expected = boolean_trace()
    lookup = pd.read_csv(ctx.results / "tables" / "gate_predictions.csv", dtype={"input_pattern": str}, float_precision="round_trip")
    lookup = lookup[lookup.gate == "NAND2"].set_index(["corner_id", "input_pattern"])
    values = lookup[["leakage_ml_A", "input_a_ml_A", "input_b_ml_A"]].to_dict("index")
    gates = {gate["name"]:gate for gate in instances()}
    primary = {f"d{i}" for i in range(8)}
    traces_by_input = {index:group.to_dict("records") for index,group in trace.groupby("input_id")}
    for corner in range(200):
        rows = paired[paired.corner_id == corner].sort_values("input_id")
        assert np.array_equal(rows.input_id, np.arange(256))
        parity = np.array([p.bit_count() % 2 for p in range(256)])
        np.testing.assert_array_equal(rows.parity_observed, parity)
        # Independently sum all actual gate instances, not only pattern counts.
        repeated_sum = []
        for index in range(256):
            total = 0.0
            for entry in traces_by_input[index]:
                gate = gates[entry["instance"]]
                pattern = entry["gate_input_pattern"]
                record = values[(corner, pattern)]
                total += record["leakage_ml_A"]
                for j, port in enumerate(("a", "b")):
                    if gate[port] not in primary and pattern[j] == "1":
                        total += record[f"input_{port}_ml_A"]
            repeated_sum.append(total)
        np.testing.assert_allclose(rows.leakage_ml_A, repeated_sum, rtol=1e-12, atol=1e-20)
    text = (ctx.netlists / "parity8.cir").read_text()
    assert sum(line.startswith("M_") for line in text.splitlines()) == 112
    assert sum(line.startswith("V_d") for line in text.splitlines()) == 8
    assert OPTIONS in text and not any(line.startswith("B") for line in text.splitlines())
    ctx.check("part6_recheck", dict(exhaustive_parity_passed=True, coverage_records=len(paired),
                                    all_instance_sums_match=True, mos_count=112, primary_inputs=8,
                                    recomputed=metrics(paired.leakage_spice_A, paired.leakage_ml_A,
                                                       ctx.config["near_zero_current_A"])))
    print("Part 6 recheck passed: exhaustive parity, physical topology, and instance-by-instance leakage sums.", flush=True)
