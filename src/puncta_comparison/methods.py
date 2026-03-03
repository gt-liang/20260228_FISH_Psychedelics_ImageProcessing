"""
Puncta Detection Methods
========================
Pure functions — each takes a 2-D nucleus image crop (spatially bg-corrected,
float64) and a boolean mask of the same shape, and returns a single float
representing the signal strength of that channel for that nucleus.

All methods are designed to be comparable via argmax: whichever channel
returns the highest signal is called as the colour for that round.

Methods
-------
X  — max pixel intensity (current primary method, no explicit detection)
Y  — adaptive threshold (mean + n*sigma), integrated intensity above threshold
Z  — Laplacian of Gaussian (LoG) blob detection, peak intensity of best blob
W  — Difference of Gaussians (DoG) blob detection, peak intensity of best blob
T  — TrackPy Crocker-Grier centroid, mass of brightest detected spot
P  — Gaussian pre-filter + peak_local_max, peak intensity of strongest peak
"""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog, blob_log, peak_local_max


# ──────────────────────────────────────────────────────────────────────────────
# Method X — Max pixel intensity
# ──────────────────────────────────────────────────────────────────────────────
def method_x(crop: np.ndarray, mask: np.ndarray, **_) -> float:
    """
    Max pixel intensity within the nucleus mask.
    Current primary method (Module 5 Method X).
    No explicit spot detection — robust to spot morphology variation.
    """
    pixels = crop[mask]
    return float(pixels.max()) if pixels.size > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Method Y — Adaptive threshold, integrated intensity
# ──────────────────────────────────────────────────────────────────────────────
def method_y(crop: np.ndarray, mask: np.ndarray,
             n_sigma: float = 6.0, **_) -> float:
    """
    Ronan-style adaptive threshold: mean + n_sigma * std over the nucleus.
    Returns the integrated (summed) intensity of pixels above threshold.

    Scientific note: integrating above-threshold pixels captures the
    total fluorescence in the punctum, not just the peak — more robust
    to single bright pixels from noise.
    If no pixel exceeds the threshold, returns 0.
    """
    pixels = crop[mask]
    if pixels.size == 0:
        return 0.0
    thresh = pixels.mean() + n_sigma * pixels.std()
    above  = pixels[pixels > thresh]
    return float(above.sum()) if above.size > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Method Z — Laplacian of Gaussian (LoG) blob detection
# ──────────────────────────────────────────────────────────────────────────────
def method_z_log(crop: np.ndarray, mask: np.ndarray,
                 min_sigma: float = 1.0, max_sigma: float = 4.0,
                 log_threshold: float = 0.02, **_) -> float:
    """
    Laplacian of Gaussian (LoG) blob detection on normalised crop.
    Returns the peak intensity (in original ADU scale) of the brightest
    detected blob that falls within the nucleus mask.

    LoG is the classic diffraction-limited spot detector — optimal for
    spots of known size (sigma ≈ spot_radius / sqrt(2)).
    At 20x with ~14 px² spots: sigma ≈ sqrt(14/pi) / sqrt(2) ≈ 1.5 px.
    """
    if crop.max() < 1:
        return 0.0

    norm = crop / crop.max()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blobs = blob_log(norm, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=10, threshold=log_threshold)

    return _best_blob_peak(blobs, crop, mask)


# ──────────────────────────────────────────────────────────────────────────────
# Method W — Difference of Gaussians (DoG) blob detection
# ──────────────────────────────────────────────────────────────────────────────
def method_w_dog(crop: np.ndarray, mask: np.ndarray,
                 min_sigma: float = 1.0, max_sigma: float = 4.0,
                 dog_threshold: float = 0.02, **_) -> float:
    """
    Difference of Gaussians (DoG) blob detection — fast approximation of LoG.
    Returns peak intensity of the brightest in-mask blob.
    """
    if crop.max() < 1:
        return 0.0

    norm = crop / crop.max()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blobs = blob_dog(norm, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=dog_threshold)

    return _best_blob_peak(blobs, crop, mask)


def _best_blob_peak(blobs: np.ndarray, crop: np.ndarray,
                    mask: np.ndarray) -> float:
    """Helper: from a blob array (y, x, sigma), return the max in-mask peak."""
    if len(blobs) == 0:
        return 0.0
    best = 0.0
    for y, x, *_ in blobs:
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
            val = float(crop[yi, xi])
            if val > best:
                best = val
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Method T — TrackPy Crocker-Grier centroid
# ──────────────────────────────────────────────────────────────────────────────
def method_t_trackpy(crop: np.ndarray, mask: np.ndarray,
                     diameter: int = 5, **_) -> float:
    """
    TrackPy Crocker-Grier centroid-finding algorithm.
    Returns the integrated mass (sum of pixel ring) of the brightest
    detected spot whose centre falls within the nucleus mask.

    diameter must be an odd integer ≥ 3.
    At 20x: spot radius ≈ 2 px → diameter = 5 px.

    Mass is comparable across channels for argmax decoding because
    it integrates brightness over the same fixed-diameter region.
    """
    try:
        import trackpy as tp
    except ImportError:
        return method_x(crop, mask)   # graceful fallback

    # TrackPy expects a 2D frame with positive intensities
    frame = np.maximum(crop, 0).astype(np.float64)
    if frame.max() < 1:
        return 0.0

    diameter = max(3, int(diameter) | 1)   # ensure odd, ≥ 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            features = tp.locate(frame, diameter=diameter,
                                 minmass=0, engine="python")
        except Exception:
            return 0.0

    if features is None or len(features) == 0:
        return 0.0

    best_mass = 0.0
    for _, row in features.iterrows():
        yi, xi = int(round(row["y"])), int(round(row["x"]))
        if (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]
                and mask[yi, xi]):
            m = float(row["mass"])
            if m > best_mass:
                best_mass = m
    return best_mass


# ──────────────────────────────────────────────────────────────────────────────
# Method P — Gaussian pre-filter + peak_local_max
# ──────────────────────────────────────────────────────────────────────────────
def method_p_peak(crop: np.ndarray, mask: np.ndarray,
                  sigma: float = 1.0, min_distance: int = 3, **_) -> float:
    """
    Gaussian smoothing (sigma=1 px) followed by peak_local_max.
    Returns the original (unsmoothed) intensity at the brightest
    detected peak position within the nucleus mask.

    Gaussian pre-filtering suppresses single-pixel shot-noise spikes
    that would otherwise dominate Method X.
    """
    filtered = gaussian_filter(crop.astype(np.float64), sigma=sigma)
    if filtered.max() < 1:
        return 0.0

    coords = peak_local_max(filtered, min_distance=min_distance,
                            exclude_border=False)
    if len(coords) == 0:
        return 0.0

    best = 0.0
    for y, x in coords:
        if mask[y, x]:
            val = float(crop[y, x])
            if val > best:
                best = val
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Registry — maps method name → callable
# ──────────────────────────────────────────────────────────────────────────────
METHODS = {
    "X":   method_x,
    "Y":   method_y,
    "Z":   method_z_log,
    "W":   method_w_dog,
    "T":   method_t_trackpy,
    "P":   method_p_peak,
}

METHOD_LABELS = {
    "X": "Method X\n(max pixel)",
    "Y": "Method Y\n(adaptive thresh)",
    "Z": "Method Z\n(LoG blob)",
    "W": "Method W\n(DoG blob)",
    "T": "Method T\n(TrackPy)",
    "P": "Method P\n(peak local max)",
}
