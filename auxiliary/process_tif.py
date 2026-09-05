import numpy as np

CONVERT_UINT16_FLOAT64 = 65535.0

def _as_float32(x):
    return np.asarray(x, dtype=np.float32)

def _safe_scale(scale, eps=1e-8):
    scale = float(scale)
    return scale if abs(scale) >= eps else 1.0

def tif2ndarray_min_max(tif_arr):
    x = _as_float32(tif_arr)
    tif_min = float(np.min(x))
    tif_max = float(np.max(x))
    scale = _safe_scale(tif_max - tif_min)
    y = (x - tif_min) / scale
    return y.astype(np.float32), tif_max, tif_min

def tif2ndarray_robust_min_max(
    tif_arr,
    lower_percentile=0.001,
    upper_percentile=99.99,
):
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError(
            f"Invalid robust percentiles: {lower_percentile}, {upper_percentile}"
        )

    x = _as_float32(tif_arr)
    tif_min = float(np.percentile(x, lower_percentile))
    tif_max = float(np.percentile(x, upper_percentile))
    scale = _safe_scale(tif_max - tif_min)

    y = np.clip(x, tif_min, tif_max)
    y = (y - tif_min) / scale

    return y.astype(np.float32), tif_max, tif_min

def tif2ndarray_mean(tif_arr):
    x = _as_float32(tif_arr)
    tif_mean = float(np.mean(x))
    return (x - tif_mean).astype(np.float32), tif_mean

def tif2ndarray_mean_std(tif_arr, eps=1e-8, out_dtype=np.float32):
    x = _as_float32(tif_arr)
    tif_mean = float(np.mean(x))
    tif_std = float(np.std(x))
    tif_std = _safe_scale(tif_std, eps=eps)
    x_norm = (x - tif_mean) / tif_std
    return x_norm.astype(out_dtype), tif_mean, tif_std

def tif2ndarray_mean_max(tif_arr):
    x = _as_float32(tif_arr)
    tif_mean = float(np.mean(x))
    tif_max_minus_mean = float(np.max(x) - tif_mean)
    tif_max_minus_mean = _safe_scale(tif_max_minus_mean)
    normalized_video = (x - tif_mean) / tif_max_minus_mean
    return normalized_video.astype(np.float32), tif_mean, tif_max_minus_mean

def apply_normalization(
    tif_arr,
    pre_norm="robust_min_max",
    norm_stats=None,
    robust_low=0.001,
    robust_high=99.99,
):
    """
    Normalize a TIFF array and return (normalized_array, norm_stats).

    norm_stats can be supplied to force test/generalization data to use the
    exact same normalization parameters as the training data.
    """
    x = _as_float32(tif_arr)

    if norm_stats is not None:
        mode = norm_stats["mode"]

        if mode in ("min_max", "robust_min_max"):
            lo = float(norm_stats["min"])
            hi = float(norm_stats["max"])
            scale = _safe_scale(hi - lo)
            if mode == "robust_min_max":
                x = np.clip(x, lo, hi)
            y = (x - lo) / scale
            return y.astype(np.float32), dict(norm_stats)

        if mode == "mean":
            mean = float(norm_stats["mean"])
            return (x - mean).astype(np.float32), dict(norm_stats)

        if mode == "mean_std":
            mean = float(norm_stats["mean"])
            std = _safe_scale(norm_stats["std"])
            return ((x - mean) / std).astype(np.float32), dict(norm_stats)

        if mode == "mean_max":
            mean = float(norm_stats["mean"])
            scale = _safe_scale(norm_stats["max_minus_mean"])
            return ((x - mean) / scale).astype(np.float32), dict(norm_stats)

        raise ValueError(f"Unsupported normalization mode in norm_stats: {mode}")

    if pre_norm == "min_max":
        y, hi, lo = tif2ndarray_min_max(x)
        stats = {"mode": "min_max", "min": lo, "max": hi}

    elif pre_norm == "robust_min_max":
        y, hi, lo = tif2ndarray_robust_min_max(
            x,
            lower_percentile=robust_low,
            upper_percentile=robust_high,
        )
        stats = {
            "mode": "robust_min_max",
            "min": lo,
            "max": hi,
            "lower_percentile": float(robust_low),
            "upper_percentile": float(robust_high),
        }

    elif pre_norm == "mean":
        y, mean = tif2ndarray_mean(x)
        stats = {"mode": "mean", "mean": mean}

    elif pre_norm == "mean_std":
        y, mean, std = tif2ndarray_mean_std(x)
        stats = {"mode": "mean_std", "mean": mean, "std": std}

    elif pre_norm == "mean_max":
        y, mean, max_minus_mean = tif2ndarray_mean_max(x)
        stats = {
            "mode": "mean_max",
            "mean": mean,
            "max_minus_mean": max_minus_mean,
        }

    else:
        raise ValueError(f"Unsupported pre_norm: {pre_norm}")

    return y.astype(np.float32), stats

def denormalize_to_uint16(np_arr, norm_stats):
    """
    Exact inverse of apply_normalization, followed only by uint16 clipping.

    Important: this function does NOT re-normalize using the reconstruction's
    own min/max. Therefore absolute intensity and trace amplitude are preserved.
    """
    x = _as_float32(np_arr)
    mode = norm_stats["mode"]

    if mode in ("min_max", "robust_min_max"):
        lo = float(norm_stats["min"])
        hi = float(norm_stats["max"])
        x = np.clip(x, 0.0, 1.0)
        y = x * (hi - lo) + lo

    elif mode == "mean":
        y = x + float(norm_stats["mean"])

    elif mode == "mean_std":
        y = x * float(norm_stats["std"]) + float(norm_stats["mean"])

    elif mode == "mean_max":
        y = (
            x * float(norm_stats["max_minus_mean"])
            + float(norm_stats["mean"])
        )

    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    return np.clip(y, 0.0, CONVERT_UINT16_FLOAT64).astype(np.uint16)
