# PVT-aware leakage estimation

The assignment uses the original `32nm_MGK.pm`, 1.0 V nominal supply, 32 nm
channel length, and an 8-bit parity checker. `P = D0 XOR ... XOR D7` is 1 for
odd parity; D0 is the least significant input bit. The formal report remains
**deferred by user request**.

## Run

Use Python 3.12, NGSpice 42, and the pinned libraries in `requirements.txt`.
On a new Ubuntu/WSL installation, install the runtime and create an environment:

```bash
sudo apt update
sudo apt install python3 python3-venv ngspice
python3 -m venv ~/.local/share/venvs/vlsi
source ~/.local/share/venvs/vlsi/bin/activate
python -m pip install -r requirements.txt
python src/main.py all
```

In the already configured WSL environment, the shared `vlsi-python` launcher is
available from any directory. From this project directory:

```bash
vlsi-python src/main.py all
```

This runs `setup`, `data`, `train`, `gates`, `predict`, `circuit`, and `audit`, in
order. Historical benchmarking is added only when its optional local files are
present. Each required stage is also available separately. To reproduce in a
clean, unused output location:

```bash
vlsi-python src/main.py all --output /tmp/parity-fresh-run
```

`vlsi-python src/main.py audit` recomputes the saved metrics and checks coverage,
splits, model reloads, current conservation, and example netlists. Run a standalone
example from its directory, e.g. `(cd netlists && ngspice -b NAND2.cir)`.

## Revision 3: four accuracy improvements

1. **Circuit supply accounting.** Gate validation still compares current drawn
   from the gate's VDD source. Circuit composition additionally counts the signed
   current supplied to each internal HIGH input, once per receiving input port.
   External primary-input sources remain separate. Thus the circuit estimate is
   `sum(gate VDD currents) + sum(internal HIGH input currents)`. Both terms use ML
   terminal-current predictions. Corresponding SPICE-current sums are separate
   diagnostics and never enter ML training or calibration. Interconnect remains
   ideal Boolean rails in the approximation; no voltage/loading solver is added.
2. **Supplementary boundary data.** The four required uniform CSVs retain 10,000
   rows and their exact schema. Four separate `_boundary.csv` files add 20,000
   transistor-only samples per case. They cover equal/near-equal drain and source,
   equal and opposite rails, one rail with an interior terminal, both orientations,
   and interior biases. Half use the gate-library width; half use continuous widths.
   Gate and bulk still obey the prescribed case rails. No gate or circuit leakage
   target is used to generate, fit, or select transistor models.
3. **Joint current prediction.** Models learn three terminal coordinates:
   `C=(Id-Is)/2`, `G=Ig`, `B=Ib`. The symmetric assigned card supports
   `C=H*(Vd-Vs)`, checked against real transistor simulations. Reconstructing
   `Id=C-(G+B)/2` and `Is=-C-(G+B)/2` enforces KCL without independently classifying
   drain/source signs. Width/bias factors and log coefficients are stored in each
   artifact. All four original signed SPICE currents remain evaluation targets;
   reconstruction errors and simulator numerical residuals are retained.
4. **Stratification and fresh evaluation.** Input-only bias regimes crossed with
   three temperature bands define fixed-seed stratified 70:15:15 splits, separately
   for uniform and boundary populations. Each case trains on 21,000 rows, validates
   on 4,500, and tests on 4,500. Population-specific test metrics remain separate.
   Seed 271828 is the final independent cohort; 314159 was development and
   20261109 belongs to the preserved original baseline.

Ridge, Extra Trees, histogram gradient boosting, and local radial-basis regression
are compared for every case (16 rows). Electrical regimes are determined from
inputs only. Each regime selects its family using transistor validation data,
and the resulting pipeline is saved as one of exactly four model artifacts.
The selection objective is equal-population mean vector relative L1 error, with
an absolute 1e-14 A denominator floor. This objective is distinct from the reported
percentage-error accuracy; it prevents numerical-zero observations from dominating
selection while retaining every output in the physical error calculation.

Local RBF models use a polynomial trend and a training-derived sensitivity metric.
Their latent coefficient estimates are bounded to nearby training support to
prevent interpolation overshoot. Numerical coefficient-fit floors are recorded in
`training_protocol.json`; they do not remove held-out records from evaluation.
One predeclared RBF refinement uses validation only. Gate/circuit evaluations
follow model freezing. See [SciPy's RBF documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).

## Conventions and checks

VDD is uniform in 0.9–1.1 V; temperature in -40–125 °C; five process factors are
independent uniform draws in [0.9, 1.1], separately for NMOS and PMOS. Uniform
transistor widths range continuously from 32–256 nm. Every gate NMOS is 64 nm,
every PMOS 128 nm, and every transistor length is 32 nm. CSV units are SI except
°C; `ndep` is m^-3 in CSV/model features and converted to cm^-3 only for SPICE.
Original card values, converted nominal values, and source hashes are retained.
The original oxide-consistency warning is preserved in simulator logs.

Terminal currents enter the MOS. Four single-device bias-source currents are
negated; gate measuring-source currents already point into their terminals.
Supply leakage is `-I(VDD)`. KCL tolerance is
`1e-14 A + 1e-6*max(|terminal currents|)`; absolute residuals are also recorded.

Percentage error is `100*abs(prediction-reference)/abs(reference)` for reference
magnitudes at least 1e-15 A. Smaller references retain absolute-error statistics
and explicit counts. **Mean accuracy is 100 minus mean percentage error, without
clipping.** Bias is mean signed error in amperes. Multi-output percentage metrics
pool eligible terminal observations; R² is uniform-average per terminal.
Per-terminal and per-population metrics prevent small-current errors being hidden.
Log plots use magnitudes, exclude zeros, and mark sign differences in red. Signed
CSVs are preserved. Large relative errors near small terminal currents remain
visible, including poor boundary-test means where applicable.

Gate ML inputs reuse recorded SPICE node voltages with explicit origin labels;
these are not independently predicted voltages. AND2/OR2 retain their actual
intermediate gate voltage and route to the nearest gate-rail case. The complete
112-MOS parity circuit remains physically connected for SPICE reference.
Temporary `.nodeset` guesses assist Newton convergence and are released before
solving; no behavioral XOR is present. See the [NGSpice manual](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf).

## Artifacts

| Formal specification | Deliverables |
| --- | --- |
| Sections 2, 4; Section 6 items 2, 9 | Configuration, original model, four transistor and five gate examples, complete circuit netlist |
| Sections 3A, 4; items 1, 3 | Uniform datasets, supplementary datasets, stratified indices, KCL checks, three trend plots |
| Section 3A; items 4, 5 | 16-row comparison, four pipelines, validation/test pairs, metrics and four plots |
| Sections 3B, 5; items 6, 7 | 200 shared corners, 3,600 gate records, 18-row table, parity plot |
| Sections 3C, 5; item 8 | All 256 patterns at all 200 corners: 51,200 circuit references and paired predictions |
| Section 6 item 9 | Ordered entry point, retained simulator decks/logs/numeric outputs, final audit |
| Section 6 item 10 | Formal report deferred until explicitly requested |

Local `baseline/` and `development/` archives are excluded from the submission.
The optional historical benchmark re-evaluates frozen baseline models on the same
new corners; its files are not needed to reproduce the required deliverables.
`results/checks/final_audit.json` records verified coverage and measured target flags.
Logs and batch outputs are generated under `scratch/` and excluded from submission.
The included `models/32nm_MGK.pm` is the byte-for-byte instructor-supplied card.
The original local TAR/ZIP copies were checked; only surrounding whitespace differs.
Reproduction uses the canonical included card when the optional archives are absent.

## Verified final results

The 200 fresh corners (final seed 271828) give these mean accuracies under the
unchanged 100-MAPE rule:

| Evaluation | Mean accuracy |
| --- | ---: |
| 3,600 gate records | **99.56%** |
| 51,200 complete-circuit records | **99.61%** |

The revised gate median/95th-percentile errors are 0.059% / 1.982%; circuit errors
are 0.251% / 1.239%. All 51,200 parity outputs and 1,433,600 internal gate outputs
pass. Both evaluations have zero supply-current sign mismatches. The maximum device
prediction KCL residual is 1.42e-22 A.

These are aggregate leakage results, not a universal 99% claim for every terminal
or pattern. NOR2 `10` and `11` have mean accuracies 98.31% and 96.00%. Uniform
transistor-test mean accuracies are 97.01%, 83.83%, 98.96%, and 84.41% in case order
NMOS ON/OFF, PMOS ON/OFF. Boundary tests retain large relative-error outliers;
PMOS OFF boundary mean accuracy is 46.62% despite its 0.224% median error. See
`results/tables/test_metrics_by_population.csv` and the saved test prediction pairs.
Running the audit also generates optional largest-error diagnostics locally.
No final-test tuning, percentage clipping, or changed exclusion threshold is used.

The complete pipeline ran in a clean output directory. Its audit replayed the
fixed-seed inputs and stratified splits, reloaded the four model artifacts, and
recomputed published metrics from saved pairs. Submission evidence is in
`final_audit.json` and `final_test_protocol.json` under `results/checks/`.
