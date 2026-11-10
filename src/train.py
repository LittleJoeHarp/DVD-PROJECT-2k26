"""Four model families per transistor case; selection uses validation data only."""
import gc
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
                              ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from common import CASES, FEATURES, TARGETS, save_json, sha256, kcl_check
from metrics import metrics, parity_plot

FAMILIES = ("Ridge", "ExtraTrees", "HistGradientBoosting", "LocalRBF")
BASE_SETTINGS = {
    "Ridge": {"alpha": .001},
    "ExtraTrees": {"n_estimators": 128, "min_samples_leaf": 1},
    "HistGradientBoosting": {"max_iter": 500, "max_leaf_nodes": 31, "learning_rate": .06,
                             "l2_regularization": .01, "min_samples_leaf": 15, "early_stopping": False},
    "LocalRBF": {"neighbors": 96, "epsilon": .7, "smoothing": 1e-7},
}


def bias_features(values):
    """Deterministic voltage differences supplement the required raw features."""
    x = np.asarray(values, dtype=float)
    vd, vs, vg, vb = (x[:, index] for index in (3, 4, 5, 6))
    return np.column_stack([x, vd-vs, np.abs(vd-vs), vg-vs, vg-vd,
                            vd-vb, vs-vb, vg-np.minimum(vd, vs), vg-np.maximum(vd, vs),
                            np.minimum(vd, vs), np.maximum(vd, vs), np.log1p(np.abs(vd-vs)/.01)])


class CurrentModel:
    """Legacy reader retained for frozen baseline artifacts; new fits use joint.py."""
    def __init__(self, family, settings, seed):
        self.family, self.settings, self.seed = family, settings, seed
        self.feature_order, self.target_order = FEATURES, TARGETS
        self.units = {"currents": "A", "voltages": "V", "w": "m", "temp": "C", "ndep": "m^-3"}
        self.feature_units = dict(vdd="V", temp="C", w="m", vd="V", vs="V", vg="V", vb="V",
                                  vth0="V", toxe="m", ndep="m^-3", u0="m^2/(V*s)", xj="m")

    def fit(self, x, y):
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(bias_features(x))
        self.poly = PolynomialFeatures(degree=2, include_bias=False) if self.family == "Ridge" else None
        if self.poly is not None:
            x = self.poly.fit_transform(x)
        self.target_scaler = StandardScaler()
        self.magnitude_floor_A = 1e-30
        transformed = self.target_scaler.fit_transform(np.log10(np.maximum(np.abs(y), self.magnitude_floor_A)))
        constructors = {"Ridge": Ridge, "RandomForest": RandomForestRegressor,
                        "ExtraTrees": ExtraTreesRegressor, "HistGradientBoosting": HistGradientBoostingRegressor}
        settings = dict(self.settings)
        if self.family != "Ridge":
            settings["random_state"] = self.seed
        if self.family in ("RandomForest", "ExtraTrees"):
            settings["n_jobs"] = 4
        estimator = constructors[self.family](**settings)
        self.regressor = MultiOutputRegressor(estimator) if self.family == "HistGradientBoosting" else estimator
        self.regressor.fit(x, transformed)
        if self.family in ("RandomForest", "ExtraTrees"):
            self.regressor.n_jobs = 1  # Deterministic order when summing tree predictions.
        self.sign_models = []
        for column in range(4):
            signs = np.sign(y[:, column]).astype(int)
            if len(np.unique(signs)) == 1:
                classifier = DummyClassifier(strategy="most_frequent")
            elif self.family == "Ridge":
                classifier = RidgeClassifier(alpha=10)
            elif self.family in ("RandomForest", "ExtraTrees"):
                constructor = RandomForestClassifier if self.family == "RandomForest" else ExtraTreesClassifier
                classifier = constructor(n_estimators=64, max_depth=18, random_state=self.seed, n_jobs=4)
            else:
                classifier = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                    learning_rate=.08, early_stopping=False, random_state=self.seed)
            classifier.fit(x, signs)
            if hasattr(classifier, "n_jobs"):
                classifier.n_jobs = 1
            self.sign_models.append(classifier)
        return self

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.loc[:, self.feature_order].to_numpy()
        x = self.scaler.transform(bias_features(x))
        if self.poly is not None:
            x = self.poly.transform(x)
        transformed = self.target_scaler.inverse_transform(self.regressor.predict(x))
        signs = np.column_stack([classifier.predict(x) for classifier in self.sign_models])
        result = signs * np.power(10.0, transformed)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite model prediction")
        return result


def selection_error(truth, prediction, origins):
    """Equal-population vector error; numerical-zero rows use the KCL noise floor."""
    relative = 100*np.sum(abs(prediction-truth), axis=1)/np.maximum(np.sum(abs(truth), axis=1), 1e-14)
    return float(np.mean([relative[origins == population].mean() for population in ("uniform", "boundary")
                          if (origins == population).any()]))


def train(ctx):
    from data import load_training_data
    from joint import JointCurrentModel, RegimeEnsemble
    comparison, terminal_metrics, test_summary, population_metrics, selection, attempts = [], [], [], [], {}, []
    validation_pairs = {}
    regime_choices, selected_validation = [], []
    threshold = ctx.config["near_zero_current_A"]
    def retain(case, family, mask, truth, prediction, origins):
        paired = pd.DataFrame(dict(case=case, family=family, row_id=np.flatnonzero(mask), population=origins))
        for j, terminal in enumerate(TARGETS):
            paired[f"spice_{terminal}"] = truth[:, j]
            paired[f"ml_{terminal}"] = prediction[:, j]
        validation_pairs[(case, family)] = paired
    protocol = dict(revision=ctx.config["revision"], families=BASE_SETTINGS, split="stratified 70:15:15 within each population",
        selection="within each input-defined regime, minimum validation equal-population vector relative L1 error; denominator floor 1e-14 A",
        target_transform="log positive symmetric conductance and gate/body coefficients; exact joint KCL reconstruction",
        coordinate_noise_floors_A=[1e-14,1e-16,1e-14], original_signed_currents_used_for_metrics=True,
        rbf_training_support_bounds=True,
        gate_circuit_targets_used=False, train_only_transforms=True, kcl_parameterization=True,
        seed=ctx.config["seed"], data_hashes={c:sha256(ctx.data/f"{c}.csv") for c in CASES},
        supplementary_hashes={c:sha256(ctx.data/f"{c}_boundary.csv") for c in CASES},
        refinement_settings=dict(BASE_SETTINGS["LocalRBF"],neighbors=160,epsilon=.5,smoothing=1e-8))
    ctx.check("training_protocol", protocol)
    with threadpool_limits(limits=2):
        for case_index, case in enumerate(CASES):
            frame, labels, origins = load_training_data(ctx, case)
            x, y = frame.loc[:,FEATURES].to_numpy(), frame.loc[:,TARGETS].to_numpy()
            training, validation = labels == "train", labels == "validation"
            best, best_score, best_settings = None, float("inf"), None
            candidates, candidate_predictions = {}, {}
            for family in FAMILIES:
                started=time.perf_counter()
                model=JointCurrentModel(family,dict(BASE_SETTINGS[family]),ctx.config["seed"]+case_index,case)
                model.fit(x[training],y[training])
                prediction=model.predict(x[validation])
                score=metrics(y[validation],prediction,threshold)
                objective=selection_error(y[validation],prediction,origins[validation])
                row=dict(case=case,family=family,configuration="baseline",**score,
                         selection_error_pct=objective,fit_predict_seconds=time.perf_counter()-started)
                comparison.append(row);attempts.append(dict(row))
                candidates[family] = model
                candidate_predictions[family] = prediction
                retain(case,family,validation,y[validation],prediction,origins[validation])
                for j,terminal in enumerate(TARGETS):
                    terminal_metrics.append(dict(case=case,family=family,split="validation",terminal=terminal,
                        **metrics(y[validation,j],prediction[:,j],threshold)))
                if objective<best_score: best,best_score,best_settings=model,objective,dict(BASE_SETTINGS[family])
                print(f"{case} {family}: validation vector error={objective:.3f}%, terminal MAPE={score['mean_error_pct']:.2f}%, {time.perf_counter()-started:.1f}s",flush=True)
                del model;gc.collect()
            # A single predeclared local-RBF refinement, evaluated on validation only.
            if best_score>.5:
                settings=protocol["refinement_settings"]
                started=time.perf_counter()
                candidate=JointCurrentModel("LocalRBF",dict(settings),ctx.config["seed"]+case_index,case)
                candidate.fit(x[training],y[training])
                prediction=candidate.predict(x[validation])
                score=metrics(y[validation],prediction,threshold)
                objective=selection_error(y[validation],prediction,origins[validation])
                attempts.append(dict(case=case,family="LocalRBF",settings=settings,**score,selection_error_pct=objective))
                baseline=next(r for r in comparison if r["case"]==case and r["family"]=="LocalRBF")
                if objective<baseline["selection_error_pct"]:
                    candidates["LocalRBF"] = candidate
                    candidate_predictions["LocalRBF"] = prediction
                    baseline.update(score,selection_error_pct=objective,configuration="refined")
                    baseline["fit_predict_seconds"]+=time.perf_counter()-started
                    retain(case,"LocalRBF",validation,y[validation],prediction,origins[validation])
                    terminal_metrics=[r for r in terminal_metrics if not(r["case"]==case and r["family"]=="LocalRBF" and r["split"]=="validation")]
                    for j,terminal in enumerate(TARGETS):
                        terminal_metrics.append(dict(case=case,family="LocalRBF",split="validation",terminal=terminal,
                            **metrics(y[validation,j],prediction[:,j],threshold)))
                if objective<best_score:best,best_score,best_settings=candidate,objective,dict(settings)
                print(f"{case} bounded RBF refinement: validation vector error={objective:.3f}%",flush=True)
                del candidate;gc.collect()
            regimes = candidates[FAMILIES[0]].regimes(x[validation])
            mapping = {}
            combined = np.empty_like(y[validation])
            for regime in np.unique(regimes):
                mask = regimes == regime
                objectives = {family:selection_error(y[validation][mask],prediction[mask],origins[validation][mask])
                              for family,prediction in candidate_predictions.items()}
                family = min(objectives, key=objectives.get)
                mapping[int(regime)] = family
                combined[mask] = candidate_predictions[family][mask]
                regime_choices.append(dict(case=case,regime=int(regime),family=family,validation_rows=int(mask.sum()),
                                            selection_error_pct=objectives[family]))
            best = RegimeEnsemble(candidates, mapping)
            best_score = selection_error(y[validation],combined,origins[validation])
            best_settings = dict(families=best.settings,region_mapping=mapping)
            selected_validation.append(dict(case=case,family=best.family,selection_error_pct=best_score,
                                            **metrics(y[validation],combined,threshold)))
            print(f"{case} validation-selected regime ensemble: vector error={best_score:.3f}%, families={mapping}",flush=True)
            selection[case]=dict(family=best.family,settings=best_settings,validation_selection_error_pct=best_score,
                training_rows=int(training.sum()),selection_used_test=False,coordinate_fit_rows=best.fitted_coordinate_rows)
            best.metadata=dict(case=case,**selection[case],dataset_sha256=protocol["data_hashes"][case],
                supplementary_sha256=protocol["supplementary_hashes"][case],length_m=ctx.config["length_m"],
                nominal_vdd=ctx.config["nominal_vdd"],model_card_sha256=sha256(ctx.models/ctx.config["model_file"]))
            path=ctx.results/"trained"/f"{case}.joblib"
            joblib.dump(best,path,compress=3)
            # Tests are inspected only after this case's model is frozen.
            test=labels=="test";prediction=best.predict(x[test])
            reloaded=joblib.load(path).predict(x[test])
            np.testing.assert_allclose(reloaded,prediction,rtol=1e-12,atol=1e-25)
            test_summary.append(dict(case=case,family=best.family,selection_error_pct=selection_error(y[test],prediction,origins[test]),
                **metrics(y[test],prediction,threshold)))
            for population in ("uniform","boundary"):
                mask=origins[test]==population
                population_metrics.append(dict(case=case,population=population,family=best.family,
                    **metrics(y[test][mask],prediction[mask],threshold)))
            paired=pd.DataFrame(dict(row_id=np.flatnonzero(test),population=origins[test]))
            fig,axes=plt.subplots(2,2,figsize=(9,8),layout="constrained");plot_checks={}
            for j,terminal in enumerate(TARGETS):
                terminal_metrics.append(dict(case=case,family=best.family,split="test",terminal=terminal,
                    **metrics(y[test,j],prediction[:,j],threshold)))
                paired[f"spice_{terminal}"]=y[test,j];paired[f"ml_{terminal}"]=prediction[:,j]
                plot_checks[terminal]=parity_plot(axes.flat[j],y[test,j],prediction[:,j],f"{case}: {terminal}")
            fig.savefig(ctx.results/"plots"/f"transistor_{case}.png",dpi=150);plt.close(fig)
            paired.to_csv(ctx.results/"tables"/f"test_predictions_{case}.csv",index=False,float_format="%.17g")
            kcl=kcl_check(prediction,ctx.config);assert kcl["scale_aware_failures"]==0
            ctx.check(f"model_{case}",dict(reload_consistent=True,plot=plot_checks,predicted_kcl=kcl,
                artifact_sha256=sha256(path),coordinate_fit_rows=best.fitted_coordinate_rows))
            del best;gc.collect()
            del candidates,candidate_predictions
    for name,rows in [("model_comparison",comparison),("terminal_metrics",terminal_metrics),
                      ("test_metrics",test_summary),("test_metrics_by_population",population_metrics),
                      ("selected_validation_metrics",selected_validation),("validation_regime_selection",regime_choices)]:
        pd.DataFrame(rows).to_csv(ctx.results/"tables"/f"{name}.csv",index=False)
    pd.concat(validation_pairs.values(),ignore_index=True).to_csv(ctx.results/"tables"/"validation_predictions.csv",index=False,float_format="%.17g")
    save_json(ctx.results/"checks"/"model_selection.json",selection)
    save_json(ctx.results/"checks"/"validation_refinements.json",attempts)
    assert len(comparison)==16 and len(list((ctx.results/"trained").glob("*.joblib")))==4
    print("Training passed: joint conserving models, 16 comparisons, stratified fresh held-out evaluations.",flush=True)
