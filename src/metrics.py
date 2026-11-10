"""Physical-current metrics, with an explicit near-zero policy."""
import numpy as np
from sklearn.metrics import r2_score


def metrics(reference, prediction, threshold=1e-15):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    assert reference.shape == prediction.shape
    assert np.isfinite(reference).all() and np.isfinite(prediction).all()
    error = prediction - reference
    absolute = np.abs(error)
    eligible = np.abs(reference) >= threshold
    pct = 100 * absolute[eligible] / np.abs(reference[eligible])
    excluded = absolute[~eligible]
    # R² is uniform-average over terminals for multi-output data.
    r2 = float(r2_score(reference, prediction)) if len(reference) > 1 else None
    return dict(n=int(reference.size), r2=r2,
                bias_A=float(error.mean()), mae_A=float(absolute.mean()),
                rmse_A=float(np.sqrt(np.mean(error ** 2))), max_absolute_error_A=float(absolute.max()),
                spice_median_A=float(np.median(reference)), ml_median_A=float(np.median(prediction)),
                median_error_pct=float(np.median(pct)) if len(pct) else None,
                p95_error_pct=float(np.percentile(pct, 95)) if len(pct) else None,
                mean_error_pct=float(np.mean(pct)) if len(pct) else None,
                mean_accuracy_pct=float(100 - np.mean(pct)) if len(pct) else None,
                percentage_count=int(eligible.sum()), near_zero_count=int((~eligible).sum()),
                near_zero_mae_A=float(excluded.mean()) if len(excluded) else None,
                near_zero_max_error_A=float(excluded.max()) if len(excluded) else None,
                sign_mismatches=int(((np.sign(reference) != np.sign(prediction)) & eligible).sum()))


def paired_columns(reference, prediction, threshold):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    eligible = np.abs(reference) >= threshold
    percentage = np.full(reference.shape, np.nan)
    percentage[eligible] = 100 * np.abs(prediction[eligible] - reference[eligible]) / np.abs(reference[eligible])
    return {"bias_A": prediction - reference, "absolute_error_A": np.abs(prediction - reference),
            "percentage_error": percentage, "percentage_eligible": eligible}


def parity_plot(ax, reference, prediction, title):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    valid = (reference != 0) & (prediction != 0)
    x, y = np.abs(reference[valid]), np.abs(prediction[valid])
    wrong_sign = np.sign(reference[valid]) != np.sign(prediction[valid])
    ax.scatter(x[~wrong_sign], y[~wrong_sign], s=7, alpha=.35, rasterized=True,
               label=f"Sign agrees ({int((~wrong_sign).sum()):,})")
    if wrong_sign.any():
        ax.scatter(x[wrong_sign], y[wrong_sign], s=7, alpha=.4, rasterized=True,
                   color="tab:red", label=f"Sign differs ({int(wrong_sign.sum()):,})")
    if len(x):
        lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lower, upper], [lower, upper], "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="|SPICE current| (A)",
           ylabel="|ML current| (A)", title=title)
    ax.grid(True, which="both", alpha=.2)
    ax.legend(fontsize=8, loc="best")
    return {"plotted": int(valid.sum()), "zero_excluded": int((~valid).sum()),
            "magnitude_plot": True, "sign_mismatches": int((np.sign(reference) != np.sign(prediction)).sum())}
