
#!/usr/bin/env python3
import re
import os
import argparse
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Iterable
from statsmodels.tsa.statespace.sarimax import SARIMAX
from numpy.testing import assert_allclose

# ============================
# ---------- helpers ----------
# ============================

def _choose_seasonality(params: Dict[str, float], tol: float = 1e-12) -> Optional[int]:
    s24 = sum(abs(v) for k, v in params.items() if k.endswith(".S.L24"))
    s48 = sum(abs(v) for k, v in params.items() if k.endswith(".S.L48"))
    has24, has48 = s24 > tol, s48 > tol
    if has24 and not has48:
        return 24
    if has48 and not has24:
        return 48
    if not has24 and not has48:
        return None
    # both nonzero (unexpected in your case) -> pick stronger
    return 24 if s24 >= s48 else 48

def _infer_orders(param_names: Iterable[str], chosen_s: Optional[int]) -> Tuple[int, int, int, int, int, bool]:
    ar_lags, ma_lags = [], []
    P = Q = 0
    has_intercept = any(n in param_names for n in ("intercept", "const", "drift"))
    for name in param_names:
        if name in ("intercept", "sigma2", "const", "drift"):
            continue
        m = re.fullmatch(r"(ar|ma)\.L(\d+)", name)
        if m:
            typ, lag = m.group(1), int(m.group(2))
            (ar_lags if typ == "ar" else ma_lags).append(lag)
            continue
        m = re.fullmatch(r"(ar|ma)\.S\.L(\d+)", name)
        if m:
            typ, s = m.group(1), int(m.group(2))
            if chosen_s is not None and s == chosen_s:
                if     typ == "ar": P += 1
                else:               Q += 1
    p = max(ar_lags) if ar_lags else 0
    q = max(ma_lags) if ma_lags else 0
    s = chosen_s or 0
    return p, q, P, Q, s, has_intercept

def build_sarimax_structure(y: pd.Series, params: Dict[str, float], d=0, D=0):
    assert isinstance(y, pd.Series) and len(y) > 0, "y must be a non-empty pandas Series"
    chosen_s = _choose_seasonality(params)
    p, q, P, Q, s, has_intercept = _infer_orders(params.keys(), chosen_s)

    # --- Assertions on chosen seasonality and orders ---
    if chosen_s is not None:
        assert chosen_s in (24, 48), f"Unexpected chosen seasonality: {chosen_s}"
        # Ensure the *other* seasonal set is (almost) zero
        # other = 48 if chosen_s == 24 else 24
        # other_mag = sum(abs(v) for k, v in params.items() if k.endswith(f".S.L{other}"))
        # assert other_mag < 1e-12, f"Only one seasonal set should be non-zero; found magnitude {other_mag} for S={other}"

    assert all(x >= 0 for x in (p, d, q, P, D, Q)), f"Orders must be non-negative, got {(p,d,q,P,D,Q)}"
    if s == 0:
        assert P == 0 and Q == 0 and D == 0, f"Seasonal orders must be zero when s=0, got (P,D,Q,s)=({P},{D},{Q},{s})"
    else:
        assert s == chosen_s, "Inconsistent seasonal period"

    trend = "c" if has_intercept else "n"

    model = SARIMAX(
        y,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # Check param_names have expected groups
    assert hasattr(model, "param_names") and isinstance(model.param_names, list) and len(model.param_names) > 0, \
        "Model must expose a non-empty param_names list."

    return model, {"order": (p, d, q), "seasonal_order": (P, D, Q, s), "trend": trend}

def pack_params_for_model(model: SARIMAX, named_params: Dict[str, float], default_sigma2: float = 1.0) -> np.ndarray:
    """
    Create the parameter vector in model.param_names order.
    Any missing name gets 0.0 (except sigma2 which falls back to default_sigma2).
    """
    assert default_sigma2 > 0, "default_sigma2 must be positive"
    vec = []
    seen = set()
    for nm in model.param_names:
        if nm in named_params:
            val = float(named_params[nm])
            assert np.isfinite(val), f"Parameter {nm} is not finite"
            vec.append(val)
            seen.add(nm)
        elif nm == "sigma2":
            val = float(named_params.get("sigma2", default_sigma2))
            assert val > 0 and np.isfinite(val), f"sigma2 must be positive and finite, got {val}"
            vec.append(val)
            seen.add("sigma2")
        else:
            vec.append(0.0)

    theta = np.array(vec, dtype=np.float64)
    assert theta.shape == (len(model.param_names),), \
        f"Packed theta shape mismatch: {theta.shape} vs ({len(model.param_names)},)"

    # No unexpected keys in named_params
    extra = [k for k in named_params.keys() if k not in model.param_names]
    print("the extra keys are: ", extra)
    print("the model param names are: ", model.param_names)

    # assert len(extra) == 0, f"named_params contains keys not used by model: {extra}"

    return theta

# ============================
# ---------- main ------------
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate traffic test series from predicted SARIMA parameters")
    parser.add_argument("--traffic_dataset_dir", type=str, required=True)
    traffic_dataset_dir = parser.parse_args().traffic_dataset_dir

    # ---------- file existence ----------
    test_dataset_sarima_params_loc = os.path.join(traffic_dataset_dir, "test_standard_parameters_predicted.npy")
    test_timeseries_loc = os.path.join(traffic_dataset_dir, "test_timeseries.npy")
    assert os.path.exists(test_dataset_sarima_params_loc), f"Missing: {test_dataset_sarima_params_loc}"
    assert os.path.exists(test_timeseries_loc), f"Missing: {test_timeseries_loc}"

    # ---------- load ----------
    test_dataset_sarima_params = np.load(test_dataset_sarima_params_loc, allow_pickle=True)
    test_timeseries = np.load(test_timeseries_loc)

    # ---------- shape checks ----------
    assert test_timeseries.ndim in (2, 3), f"test_timeseries shape must be (N,T) or (N,C,T); got {test_timeseries.shape}"
    if test_timeseries.ndim == 2:
        # promote to (N,1,T)
        test_timeseries = test_timeseries[:, None, :]
    N, C, T = test_timeseries.shape
    assert N > 0 and C > 0 and T > 1, f"Invalid shape for time series: {test_timeseries.shape}"
    assert len(test_dataset_sarima_params) == N, \
        f"Parameter rows ({len(test_dataset_sarima_params)}) must match N ({N})"

    generated_samples = []
    generated_ci = []

    # ----------- per-sample loop -----------
    for sidx in range(N):
        # --- inputs and basic assertions ---
        vals = test_dataset_sarima_params[sidx]
        assert isinstance(vals, (list, np.ndarray)), f"Params for sample {sidx} must be list/np.ndarray"
        names = [
            'ma.S.L48','ma.L1','ar.L5','ma.S.L24','intercept','ma.L5','ar.L2','sigma2',
            'ar.L3','ar.L4','ma.L3','ar.L1','ar.S.L48','ma.L2','ma.L4','ar.S.L24'
        ]
        assert len(names) == len(vals), f"names and vals length mismatch: {len(names)} vs {len(vals)}"
        sarima_params = dict(zip(names, vals))

        # Confirm only one seasonal set is materially non-zero (or neither)
        s24_mag = sum(abs(float(sarima_params.get(k, 0.0))) for k in sarima_params if k.endswith(".S.L24"))
        s48_mag = sum(abs(float(sarima_params.get(k, 0.0))) for k in sarima_params if k.endswith(".S.L48"))
        # assert not (s24_mag > 1e-12 and s48_mag > 1e-12), \
        #     f"Sample {sidx}: both S=24 and S=48 seasonal params are non-zero (magnitudes {s24_mag}, {s48_mag})"

        # --- series prep ---
        input_length = 1
        actual_length = T
        assert 1 <= input_length < actual_length, "input_length must be >=1 and < T"

        idx = pd.date_range("2025-01-01", periods=input_length, freq="H")
        assert len(idx) == input_length, "Date index length mismatch"

        # build dummy series but fill first `input_length` with data
        y = pd.Series(np.zeros(len(idx), dtype=float), index=idx)
        # ensure finite
        assert np.isfinite(test_timeseries[sidx, 0, :input_length]).all(), "Non-finite observed values"
        y[idx[0]:idx[-1]] = test_timeseries[sidx, 0, :input_length]

        # --- model struct ---
        model, meta = build_sarimax_structure(y, sarima_params, d=0, D=0)
        assert "order" in meta and "seasonal_order" in meta and "trend" in meta, "Meta keys missing"
        (p, d, q) = meta["order"]
        (P, D, Q, s) = meta["seasonal_order"]
        assert all(isinstance(x, int) for x in (p, d, q, P, D, Q, s))

        # --- pack params ---
        theta = pack_params_for_model(model, sarima_params, default_sigma2=1.0)
        assert theta.ndim == 1 and theta.size == len(model.param_names), \
            "Theta dimensionality mismatch"
        assert np.isfinite(theta).all(), "Theta contains non-finite values"
        assert theta[-1] > 0, "Last param often sigma2; expected positive"

        # --- filter with theta ---
        res = model.filter(theta)
        # res should have forecasts available
        assert hasattr(res, "get_forecast"), "Filtered result missing get_forecast()"
        # check the copied params round-trip
        assert hasattr(res, "params") and res.params.shape == theta.shape, "Filtered result param shape mismatch"
        # params equality (up to tolerance)
        assert_allclose(res.params, theta, rtol=1e-5, atol=1e-6, err_msg="Filtered params differ from provided theta")

        # --- forecast horizon ---
        steps = actual_length - input_length
        assert steps > 0, "Forecast steps must be positive"

        # --- forecast + CI ---
        # Note: alpha=0.01 -> 99% CI; if you intended 95%, use alpha=0.05
        pred_res = res.get_forecast(steps=steps)
        assert hasattr(pred_res, "predicted_mean"), "No predicted_mean in forecast result"
        new_mean = pred_res.predicted_mean
        assert isinstance(new_mean, (np.ndarray, pd.Series)), "predicted_mean must be array-like"
        assert len(new_mean) == steps, f"Forecast length mismatch: {len(new_mean)} vs {steps}"
        assert np.isfinite(np.asarray(new_mean)).all(), "Forecast contains non-finite values"

        new_ci = pred_res.conf_int(alpha=0.01)   # 99% CI for stricter bounds
        assert isinstance(new_ci, pd.DataFrame) and new_ci.shape[1] == 2, \
            f"conf_int must return a DataFrame with two columns (lower/upper), got {type(new_ci)} with shape {new_ci.shape}"
        assert new_ci.shape[0] == steps, "CI row count must equal forecast horizon"
        assert np.isfinite(new_ci.to_numpy()).all(), "CI contains non-finite values"
        # lower <= upper
        assert (new_ci.iloc[:, 0] <= new_ci.iloc[:, 1]).all(), "CI lower bound exceeds upper bound somewhere"

        # --- stitch generated sample ---
        observed = test_timeseries[sidx, 0, :input_length]
        assert observed.ndim == 1 and len(observed) == input_length
        generated_sample = np.concatenate([observed, np.asarray(new_mean, dtype=float)], axis=0)
        assert generated_sample.shape == (actual_length,), "Generated sample length must equal T"
        assert np.isfinite(generated_sample).all(), "Generated sample contains non-finite values"

        generated_samples.append(generated_sample)
        generated_ci.append(new_ci.to_numpy(dtype=float))

    # ---------- stack outputs ----------
    generated_samples = np.stack(generated_samples, axis=0)  # (N, T)
    assert generated_samples.shape == (N, T)
    generated_samples = np.expand_dims(generated_samples, axis=1)  # (N, 1, T)
    assert generated_samples.shape == (N, 1, T)

    generated_ci = np.stack(generated_ci, axis=0)  # (N, steps, 2)
    assert generated_ci.shape == (N, T - 1, 2)

    # ---------- save ----------
    out_dir = traffic_dataset_dir
    np.save(os.path.join(out_dir, "test_series_simulated_L48.npy"), generated_samples)
    np.save(os.path.join(out_dir, "test_series_simulated_L48_ci.npy"), generated_ci)

    print("Saved:")
    print("  ", os.path.join(out_dir, "test_series_simulated_L48.npy"))
    print("  ", os.path.join(out_dir, "test_series_simulated_L48_ci.npy"))

    # ---------- final MAE (use correct channel dim) ----------
    # Compare the forecasted part (from input_length onward)
    mae = np.abs(generated_samples[:, 0, 1:] - test_timeseries[:, 0, 1:])
    assert mae.shape == (N, T - 1)
    assert np.isfinite(mae).all(), "MAE contains non-finite values"

    print(f"MAE mean: {float(np.mean(mae)):.6f}")
    print(f"MAE std:  {float(np.std(mae)):.6f}")
    print(f"MAE min:  {float(np.min(mae)):.6f}")
    print(f"MAE max:  {float(np.max(mae)):.6f}")
