"""Joint, drain/source-symmetric terminal-current regression with exact KCL."""
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import FEATURES, TARGETS


class SmoothRBF:
    """A polynomial trend with a local RBF residual; bounded-memory interpolation."""
    def __init__(self, neighbors=96, epsilon=.7, smoothing=1e-7):
        self.neighbors, self.epsilon, self.smoothing = neighbors, epsilon, smoothing

    def fit(self, x, y):
        self.polynomial = PolynomialFeatures(degree=3, include_bias=False)
        poly = self.polynomial.fit_transform(x)
        self.trend = Ridge(alpha=1e-4).fit(poly, y)
        residual = y - self.trend.predict(poly)
        # Learn a distance metric from training-only sensitivities in independent
        # physical coordinates; derived overdrive features are excluded here.
        self.distance_columns = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        self.distance_scaler = StandardScaler().fit(x[:, self.distance_columns])
        distance = self.distance_scaler.transform(x[:, self.distance_columns])
        metric_poly = PolynomialFeatures(degree=2, include_bias=False)
        metric_trend = Ridge(alpha=.01).fit(metric_poly.fit_transform(distance), y)
        sample = distance[np.linspace(0, len(distance)-1, min(128, len(distance))).astype(int)].copy()
        sensitivity = []
        for j in range(distance.shape[1]):
            plus, minus = sample.copy(), sample.copy()
            plus[:, j] += .05; minus[:, j] -= .05
            derivative = (metric_trend.predict(metric_poly.transform(plus))-
                          metric_trend.predict(metric_poly.transform(minus)))/.1
            sensitivity.append(np.sqrt(np.mean(derivative**2)))
        sensitivity = np.asarray(sensitivity)
        self.distance_weights = np.maximum(sensitivity/max(sensitivity.max(), 1e-8), .05)
        coordinates = distance*self.distance_weights
        self.support_tree = cKDTree(coordinates)
        self.support_targets = np.asarray(y)
        self.support_neighbors = min(self.neighbors, len(x))
        self.rbf = RBFInterpolator(coordinates, residual, neighbors=self.support_neighbors,
            kernel="multiquadric", degree=0, epsilon=self.epsilon, smoothing=self.smoothing)
        return self

    def predict(self, x):
        distance = self.distance_scaler.transform(x[:, self.distance_columns])*self.distance_weights
        prediction = self.trend.predict(self.polynomial.transform(x)) + self.rbf(distance)
        # Local RBF weights can overshoot far beyond nearby observed coefficients.
        # Bound the latent log-coefficient to training-only neighbor support.
        # Older development artifacts remain loadable with their original behavior.
        if hasattr(self, "support_tree"):
            _, neighbors = self.support_tree.query(distance, k=self.support_neighbors, workers=1)
            values = self.support_targets[neighbors]
            prediction = np.clip(prediction, values.min(axis=1), values.max(axis=1))
        return prediction


class RegimeRBF:
    """Smooth local fits inside input-defined electrical regimes, with a fallback."""
    def __init__(self, **settings):
        self.settings = settings

    def fit(self, x, y, regimes):
        self.fallback = SmoothRBF(**self.settings).fit(x, y)
        self.models = {}
        for regime in np.unique(regimes):
            mask = regimes == regime
            if mask.sum() >= 150:
                self.models[int(regime)] = SmoothRBF(**self.settings).fit(x[mask], y[mask])
        return self

    def predict(self, x, regimes):
        prediction = np.empty(len(x))
        for regime in np.unique(regimes):
            mask = regimes == regime
            model = self.models.get(int(regime), self.fallback)
            prediction[mask] = model.predict(x[mask])
        return prediction


class JointCurrentModel:
    """Learn three terminal-current coordinates and reconstruct four conserving outputs.

    C=(Id-Is)/2, G=Ig, B=Ib. Drain/source exchange gives C=H*(Vd-Vs)
    for this symmetric card. Id=C-(G+B)/2 and Is=-C-(G+B)/2 enforce KCL.
    H, |G|, |B| are fitted jointly through this parameterization, rather than
    independently classifying Id/Is signs. Original signed outputs remain the
    evaluation targets, including simulator KCL residuals and zero-bias noise.
    """
    def __init__(self, family, settings, seed, case):
        self.family, self.settings, self.seed, self.case = family, settings, seed, case
        self.feature_order, self.target_order = FEATURES, TARGETS
        self.units = dict(currents="A", voltages="V", w="m", temp="C", ndep="m^-3")
        self.feature_units = dict(vdd="V", temp="C", w="m", vd="V", vs="V", vg="V", vb="V",
                                  vth0="V", toxe="m", ndep="m^-3", u0="m^2/(V*s)", xj="m")
        self.polarity = -1 if case.startswith("pmos") else 1
        self.gate_sign = 1 if case in ("nmos_on", "pmos_off") else -1
        self.body_sign = self.polarity * -1

    def features(self, values):
        x = np.asarray(values, dtype=float)
        vdd, temp, w, vd, vs, vg, vb, vth, tox, ndep, u0, xj = x.T
        d, s, g = self.polarity*(vd-vb), self.polarity*(vs-vb), self.polarity*(vg-vb)
        low, gap = np.minimum(d, s), abs(d-s)
        thermal = 8.617333262145e-5 * (temp+273.15)
        overdrive = (g-low-abs(vth))/thermal
        return np.column_stack([vdd, 1000/(temp+273.15), np.log(w/1e-7), low/vdd, gap/vdd,
            g/vdd, abs(vth), np.log(tox/1e-9), np.log(ndep/1e24), np.log(u0), np.log(xj/1e-8),
            overdrive, np.logaddexp(0, overdrive)])

    @staticmethod
    def factors(x):
        vdd, _, w, vd, vs, vg, vb = np.asarray(x)[:, :7].T
        geometry = w/1e-7
        gate = (abs(vg-vd)+abs(vg-vs)+abs(vg-vb))/vdd
        body = (abs(vb-vd)+abs(vb-vs)+abs(vb-vg))/vdd
        return geometry, vd-vs, gate, body

    def regimes(self, x):
        f = self.features(x)
        low, gap = f[:, 3], f[:, 4]
        return np.select([(low<.04)&(gap<.04), (low>.96)&(gap<.04), gap>.94,
                          low<.04, low+gap>.96], [0,1,2,3,4], default=5).astype(int)

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        self.scaler = StandardScaler().fit(self.features(x))
        # Keep tiny gate-rail excursions from dominating the distance metric.
        self.scaler.scale_ = np.maximum(self.scaler.scale_, .05)
        features = self.scaler.transform(self.features(x))
        regimes = self.regimes(x)
        geometry, delta, gate_factor, body_factor = self.factors(x)
        c = (y[:, 0]-y[:, 2])/2
        numerators = [c, self.gate_sign*y[:, 1], self.body_sign*y[:, 3]]
        denominators = [geometry*delta, geometry*gate_factor, geometry*body_factor]
        self.coordinate_noise_floors_A = [1e-14, 1e-16, 1e-14]
        masks = [(abs(delta) >= 1e-5) & (c*delta > 0) & (abs(c) > 1e-14),
                 (gate_factor > 1e-8) & (numerators[1] > 1e-16),
                 (body_factor > 1e-8) & (numerators[2] > 1e-14)]
        # Contradictions larger than the numerical KCL tolerance invalidate the
        # sign/symmetry assumptions instead of being silently filtered out.
        assert not ((c*delta < 0) & (abs(c) > 1e-13)).any()
        assert not ((numerators[1] < 0) & (abs(y[:, 1]) > 1e-14)).any()
        assert not ((numerators[2] < 0) & (abs(y[:, 3]) > 1e-14)).any()
        targets = np.full((len(x), 3), np.nan)
        for j in range(3):
            targets[masks[j], j] = np.log10(numerators[j][masks[j]] / denominators[j][masks[j]])
        self.target_scaler = StandardScaler().fit(targets)
        standardized = self.target_scaler.transform(targets)
        self.regressors = []
        self.polynomials = []
        self.fitted_coordinate_rows = []
        for j in range(3):
            subset = features[masks[j]]
            polynomial = None
            if self.family == "Ridge":
                polynomial = PolynomialFeatures(degree=3, include_bias=False)
                subset = polynomial.fit_transform(subset)
                estimator = Ridge(**self.settings)
            elif self.family == "ExtraTrees":
                estimator = ExtraTreesRegressor(**self.settings, random_state=self.seed, n_jobs=4)
            elif self.family == "HistGradientBoosting":
                estimator = HistGradientBoostingRegressor(**self.settings, random_state=self.seed)
            elif self.family == "LocalRBF":
                estimator = RegimeRBF(**self.settings)
            else:
                raise ValueError(self.family)
            if self.family == "LocalRBF":
                estimator.fit(subset, standardized[masks[j], j], regimes[masks[j]])
            else:
                estimator.fit(subset, standardized[masks[j], j])
            if hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1
            self.regressors.append(estimator)
            self.polynomials.append(polynomial)
            self.fitted_coordinate_rows.append(int(masks[j].sum()))
        return self

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.loc[:, FEATURES].to_numpy()
        x = np.asarray(x, dtype=float)
        features = self.scaler.transform(self.features(x))
        regimes = self.regimes(x)
        coordinates = np.empty((len(x), 3))
        for j, estimator in enumerate(self.regressors):
            values = features if self.polynomials[j] is None else self.polynomials[j].transform(features)
            coordinates[:, j] = estimator.predict(values, regimes) if self.family == "LocalRBF" else estimator.predict(values)
        physical = np.power(10.0, self.target_scaler.inverse_transform(coordinates))
        geometry, delta, gate_factor, body_factor = self.factors(x)
        c = physical[:, 0] * geometry * delta
        gate = self.gate_sign * physical[:, 1] * geometry * gate_factor
        body = self.body_sign * physical[:, 2] * geometry * body_factor
        shared = (gate+body)/2
        result = np.column_stack([c-shared, gate, -c-shared, body])
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite joint-model prediction")
        return result


class RegimeEnsemble:
    """One four-current artifact; each bias regime uses a validation-selected family."""
    def __init__(self, candidates, mapping):
        self.mapping = {int(key):value for key,value in mapping.items()}
        self.models = {name:candidates[name] for name in sorted(set(mapping.values()))}
        representative = next(iter(self.models.values()))
        for attribute in ("feature_order", "target_order", "units", "feature_units", "scaler",
                          "target_scaler", "fitted_coordinate_rows", "case"):
            setattr(self, attribute, getattr(representative, attribute))
        self.family = "RegimeEnsemble"
        self.settings = {name:model.settings for name,model in self.models.items()}

    def regimes(self, x):
        return next(iter(self.models.values())).regimes(x)

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.loc[:, FEATURES].to_numpy()
        x = np.asarray(x)
        representative = next(iter(self.models.values()))
        regimes = representative.regimes(x)
        prediction = np.empty((len(x), 4))
        for regime in np.unique(regimes):
            mask = regimes == regime
            prediction[mask] = self.models[self.mapping[int(regime)]].predict(x[mask])
        return prediction
