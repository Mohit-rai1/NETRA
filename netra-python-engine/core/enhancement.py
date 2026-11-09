"""
Project NETRA — Advanced Physics-Guided Underwater Image Enhancement Pipeline
==============================================================================

Background — Why underwater images degrade
-------------------------------------------
Water absorbs and scatters light unevenly across wavelengths:
  · Red channel attenuates most aggressively — often gone by ~5 m depth.
  · Blue/green channels dominate, producing the characteristic blue-green cast.
  · Suspended particles backscatter light, reducing contrast and introducing haze.

Pipeline Overview (Phase 1 — Classical Physics-Guided CV)
---------------------------------------------------------
  Step 1  Guided Color Balancing
          A channel-mean equalisation that is gated by the green channel (g)
          so that dark shadow regions are not flooded with artificial red.
          A damped Gray-World final pass (gains clamped to [0.75, 1.25])
          prevents open-water scenes from being pushed into neutral grey.

  Step 2  Morphological Pyramid Reconstruction (MPR) on Luminance
          Decomposes the L* channel (CIE Lab) into a low-frequency (Gaussian
          blur) base and a high-frequency residual.  The M-operator isolates
          and preserves morphologically significant detail above threshold t,
          and the contrast amplification factor c=2.4 boosts local structure
          without introducing ringing artefacts.

  Step 3  Normalized Unsharp Masking (on L* only)
          Blends the original L* with a percentile-stretched high-pass to
          sharpen detail without amplifying colour-channel skew.

  Step 4  Multiscale Laplacian–Gaussian Pyramid Fusion
          Combines the MPR output (in1) and sharpened output (in2) using
          per-pixel, per-scale weights derived from Laplacian energy,
          saliency, and saturation maps.  Pyramid fusion avoids ghosting
          and block artefacts common in flat-weight blending.

  Step 5  Luminance-Preserving Adaptive Gamma Correction
          Estimates a scene-dependent gamma from the mean L* value and
          applies it in Lab space (L channel only), so colour saturation
          is unaffected.

Phase 2 (stubbed) — PyTorch Deep-Learning Model
------------------------------------------------
  torch_enhance() is a fully documented stub.  Replace its body with
  a forward pass through a trained model (Water-Net, UGAN, etc.) when
  weights become available.  The function signature is frozen.

Public API
----------
  run_pipeline(raw_bytes: bytes) -> EnhancementResult
      Single entry point called by the FastAPI endpoint in main.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import reconstruction

from config import settings

logger = logging.getLogger("netra.enhancement")


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

PARAMS: dict = {
    "sigma": 15,        # Gaussian low-pass filter sigma
    "t": 0.001,         # Morphological amplitude threshold
    "s": 0.1,           # Region size threshold coefficient
    "c": 2.4,           # Contrast control coefficient (MPR amplification)
    "alpha": 0.8,       # Color compensation strength
    "delta": 0.1,       # Weight regularization term (prevents zero weights)
    "area_scale": 100,  # min_area = max(1, round(s * area_scale))
}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnhancementResult:
    """
    Immutable container returned by :func:`run_pipeline`.

    Attributes
    ----------
    image_bgr:
        The processed image as a NumPy ``uint8`` array in OpenCV BGR format,
        shape ``(H, W, 3)``.  Callers (FastAPI endpoint) encode this to JPEG/PNG.
    steps_applied:
        Ordered list of step labels that were executed, used for the
        ``X-NETRA-Steps`` response header and server-side observability.
    """

    image_bgr: np.ndarray
    steps_applied: list[str]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _stretch(x: np.ndarray) -> np.ndarray:
    """
    Percentile-based linear stretch to [0, 1].

    Uses the 0.5th and 99.5th percentiles as lo/hi anchors so that
    hot-pixel outliers and sensor watermarks do not pin the stretch
    at extreme values and wash out the rest of the image.

    Parameters
    ----------
    x : np.ndarray
        Arbitrary-range float array.

    Returns
    -------
    np.ndarray
        Values clipped and linearly rescaled to ``[0, 1]``.
    """
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _luminance(img: np.ndarray) -> np.ndarray:
    """
    Compute perceptual luminance from an RGB float image.

    Uses the ITU-R BT.601 luma coefficients (same as OpenCV's RGB→Gray).

    Parameters
    ----------
    img : np.ndarray
        RGB float32 image, values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        2-D float32 luminance map, shape ``(H, W)``.
    """
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


# ---------------------------------------------------------------------------
# Step 1 — Guided Color Balancing
# ---------------------------------------------------------------------------

def color_balance(img: np.ndarray, alpha: float = PARAMS["alpha"]) -> np.ndarray:
    """
    Apply physics-guided channel-mean equalisation to an underwater RGB image.

    **Why not plain Gray World?**
    Plain Gray World divides each channel uniformly by its mean, which
    floods dark shadow regions with large artificial red gains, producing
    a "red shadow" artefact.  Here the correction is *gated by the green
    channel* (``g``): since ``g`` is close to 0 in shadows, the correction
    naturally tapers off in dark areas.

    **Algorithm**
    1. Compute per-channel means ``(mr, mg, mb)``.
    2. Boost red if it is below green: ``rc = r + α·max(0, mg−mr)·(1−r)·g``.
    3. Boost blue if it is below green: ``bc = b + α·max(0, mg−mb)·(1−b)·g``.
    4. Apply a *damped* Gray-World pass (gains clamped to ``[0.75, 1.25]``)
       so that fully blue water scenes are not pushed into neutral grey.

    Parameters
    ----------
    img : np.ndarray
        Input RGB float32 image, shape ``(H, W, 3)``, values in ``[0, 1]``.
    alpha : float
        Compensation strength.  ``0`` = no compensation; ``1`` = full boost.
        Default: ``0.8`` (from PARAMS).

    Returns
    -------
    np.ndarray
        Colour-balanced RGB float32 image, clipped to ``[0, 1]``.
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mr, mg, mb = float(r.mean()), float(g.mean()), float(b.mean())

    # Gated red compensation — dark areas (g≈0) get near-zero boost
    rc = r + alpha * max(0.0, mg - mr) * (1.0 - r) * g

    # Blue compensation — only when blue is deficient relative to green
    bc = b + alpha * max(0.0, mg - mb) * (1.0 - b) * g

    comp = np.clip(np.dstack([rc, g, bc]), 0.0, 1.0)

    # Damped Gray-World: bring channel means closer to overall mean,
    # but clamp gains to [0.75, 1.25] to avoid over-correction in
    # open-water (high-blue) scenes.
    means = comp.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    gains = np.clip(gray / (means + 1e-8), 0.75, 1.25)
    balanced = comp * gains[None, None, :]

    logger.debug(
        "color_balance: input means R=%.3f G=%.3f B=%.3f → gains R=%.3f G=%.3f B=%.3f",
        mr, mg, mb, gains[0], gains[1], gains[2],
    )
    return np.clip(balanced, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 — Morphological Pyramid Reconstruction (MPR) on Luminance
# ---------------------------------------------------------------------------

def _area_open(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected foreground components smaller than ``min_area`` pixels.

    Parameters
    ----------
    mask : np.ndarray
        Boolean 2-D array.
    min_area : int
        Minimum component size to retain (pixels).

    Returns
    -------
    np.ndarray
        Boolean 2-D array with small components zeroed out.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _m_operator(
    residual: np.ndarray,
    t: float,
    min_area: int,
) -> np.ndarray:
    """
    Apply the morphological M-operator to a residual map.

    Thresholds the residual at ``t``, removes small connected components
    (< ``min_area`` px), then uses morphological reconstruction by dilation
    to smoothly propagate significant extrema back into the residual.

    Parameters
    ----------
    residual : np.ndarray
        Positive-only float residual map (shape ``(H, W)``).
    t : float
        Minimum amplitude threshold — residual values below ``t`` are
        considered noise and suppressed.
    min_area : int
        Minimum connected component area to retain.

    Returns
    -------
    np.ndarray
        Reconstructed residual map of the same shape.
    """
    mask = residual >= t
    if min_area > 1:
        mask = _area_open(mask, min_area)
    marker = np.where(mask, residual, residual.min())
    return reconstruction(marker, residual, method="dilation")


def mpr(
    img: np.ndarray,
    t: float = PARAMS["t"],
    s: float = PARAMS["s"],
    c: float = PARAMS["c"],
    sigma: float = PARAMS["sigma"],
) -> np.ndarray:
    """
    Morphological Pyramid Reconstruction contrast enhancement.

    Operates exclusively on the **L*** channel (CIE Lab) to avoid
    colour-channel crosstalk and the red over-amplification that occurs
    when contrast boost is applied per-RGB-channel.

    **Algorithm**
    1. Convert RGB → CIE Lab; normalise L* to ``[0, 1]``.
    2. Compute a Gaussian low-pass base ``low`` (sigma=``sigma``).
    3. Split the detail residual (``L* − low``) into positive and negative parts.
    4. Apply the M-operator to each part independently, isolating morphologically
       significant detail above threshold ``t`` and suppressing noise/small blobs.
    5. Reconstruct: ``L*_out = low + c·(M_pos − M_neg)``; c=2.4 provides strong
       but artefact-free contrast amplification.
    6. Convert back Lab → RGB.

    Parameters
    ----------
    img : np.ndarray
        Input RGB float32 image, ``[0, 1]``.
    t : float
        Morphological amplitude threshold.
    s : float
        Region size coefficient; ``min_area = max(1, round(s × area_scale))``.
    c : float
        Contrast amplification factor.
    sigma : float
        Gaussian low-pass sigma.

    Returns
    -------
    np.ndarray
        Contrast-enhanced RGB float32 image, ``[0, 1]``.
    """
    min_area = max(1, int(round(s * PARAMS["area_scale"])))

    # RGB float32 → CIE Lab float32 (L in [0,100], a,b in [-127,127])
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    u = lab[..., 0] / 100.0  # Normalise L* → [0, 1]

    # Low-frequency base
    low = cv2.GaussianBlur(u, (0, 0), sigma)

    # Signed residual → split into positive and negative halves
    res = u - low
    res_pos = np.maximum(res, 0.0)
    res_neg = np.maximum(-res, 0.0)

    # M-operator on each half
    m_pos = _m_operator(res_pos, t, min_area)
    m_neg = _m_operator(res_neg, t, min_area)

    # Reconstruct enhanced L* channel
    l_out = np.clip(low + c * (m_pos - m_neg), 0.0, 1.0)
    lab[..., 0] = (l_out * 100.0).astype(np.float32)

    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    logger.debug("mpr: c=%.2f, t=%.4f, sigma=%.1f, min_area=%d", c, t, sigma, min_area)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — Normalized Unsharp Masking (L* channel only)
# ---------------------------------------------------------------------------

def normalized_unsharp(
    img: np.ndarray,
    sigma: float = PARAMS["sigma"] / 3.0,
) -> np.ndarray:
    """
    Sharpen image detail via unsharp masking applied to the L* channel.

    **Why L* only?**
    Applying unsharp masking to individual RGB channels amplifies channel
    imbalances and shifts hue near edges.  Working in Lab confines the
    sharpening to luminance while preserving colour fidelity.

    **Algorithm**
    1. Convert RGB → Lab; extract L* normalised to ``[0, 1]``.
    2. Compute high-pass = L* − Gaussian(L*); apply percentile stretch
       so hot pixels don't clip the enhancement.
    3. Blend: ``L*_sharp = (L* + high_pass) / 2``.
    4. Reinsert, convert Lab → RGB.

    Parameters
    ----------
    img : np.ndarray
        Input RGB float32 image, ``[0, 1]``.
    sigma : float
        Gaussian blur sigma for the low-pass component.

    Returns
    -------
    np.ndarray
        Sharpened RGB float32 image, ``[0, 1]``.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l_chan = lab[..., 0] / 100.0

    blur = cv2.GaussianBlur(l_chan, (0, 0), sigma)
    high_pass = _stretch(l_chan - blur)
    l_sharp = np.clip((l_chan + high_pass) / 2.0, 0.0, 1.0)

    lab[..., 0] = (l_sharp * 100.0).astype(np.float32)
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    logger.debug("normalized_unsharp: sigma=%.2f", sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 — Weight Maps & Multiscale Laplacian–Gaussian Pyramid Fusion
# ---------------------------------------------------------------------------

def _weight_laplacian(img: np.ndarray) -> np.ndarray:
    """Laplacian energy map — highlights edges and fine texture."""
    return np.abs(cv2.Laplacian(_luminance(img), cv2.CV_32F, ksize=3))


def _weight_saliency(img: np.ndarray) -> np.ndarray:
    """
    Global contrast saliency — measures per-pixel distance from the
    mean Lab colour.  High values correspond to visually prominent regions.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    blur = cv2.GaussianBlur(lab, (5, 5), 0)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    return np.sum((blur - mean_lab[None, None, :]) ** 2, axis=2).astype(np.float32)


def _weight_saturation(img: np.ndarray) -> np.ndarray:
    """
    Saturation map — standard deviation of RGB values per pixel
    relative to luminance.  Colourful regions score higher, which
    biases fusion toward the colour-corrected input.
    """
    lum = _luminance(img)[..., None]
    return np.sqrt(np.mean((img - lum) ** 2, axis=2))


def _compute_aggregated_weights(img: np.ndarray) -> np.ndarray:
    """
    Sum of three individually stretched weight maps:
    Laplacian (sharpness) + Saliency (prominence) + Saturation (colour richness).
    """
    return (
        _stretch(_weight_laplacian(img))
        + _stretch(_weight_saliency(img))
        + _stretch(_weight_saturation(img))
    )


def _normalize_weights(
    weight_list: list[np.ndarray],
    delta: float = PARAMS["delta"],
) -> list[np.ndarray]:
    """
    Normalise a list of weight maps so they sum to 1 at every pixel.

    The regularisation term ``delta`` prevents any input from receiving
    zero weight even when all three individual maps are zero (e.g. in a
    perfectly uniform patch).

    Parameters
    ----------
    weight_list : list[np.ndarray]
        Per-input weight maps (same shape).
    delta : float
        Regularisation constant added to each weight before normalisation.

    Returns
    -------
    list[np.ndarray]
        Normalised weight maps summing to 1 pixel-wise.
    """
    total = sum(weight_list) + len(weight_list) * delta
    return [(w + delta) / total for w in weight_list]


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a ``levels``-deep Gaussian pyramid by repeated pyrDown."""
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """
    Build a Laplacian pyramid from a Gaussian pyramid.

    Each level is the difference between a Gaussian level and the upsampled
    next-coarser level.  The coarsest level is copied from the Gaussian pyramid.
    """
    g_pyr = _build_gaussian_pyramid(img, levels)
    lap_pyr = []
    for i in range(levels - 1):
        h, w = g_pyr[i].shape[:2]
        up = cv2.pyrUp(g_pyr[i + 1], dstsize=(w, h))
        lap_pyr.append(g_pyr[i] - up)
    lap_pyr.append(g_pyr[-1])
    return lap_pyr


def multiscale_fusion(
    inputs: list[np.ndarray],
    weights: list[np.ndarray],
) -> np.ndarray:
    """
    Fuse multiple enhanced images using Laplacian–Gaussian pyramid blending.

    At each pyramid level, the contribution of each input is weighted by the
    corresponding Gaussian pyramid of its weight map.  This produces seamless,
    halo-free blending that respects both fine texture (high pyramid levels)
    and broad structure (coarse levels).

    Parameters
    ----------
    inputs : list[np.ndarray]
        List of RGB float32 images to fuse (same spatial dimensions).
    weights : list[np.ndarray]
        Per-input normalised weight maps (2-D, same H×W as inputs).

    Returns
    -------
    np.ndarray
        Fused RGB float32 image, clipped to ``[0, 1]``.
    """
    h, w = inputs[0].shape[:2]
    # Number of pyramid levels: bounded by [2, 7] and by the smaller image dimension
    levels = int(np.clip(np.floor(np.log2(min(h, w))) - 3, 2, 7))

    lap_inputs = [_build_laplacian_pyramid(inp, levels) for inp in inputs]
    gau_weights = [_build_gaussian_pyramid(wm, levels) for wm in weights]

    # Weighted sum at each pyramid level
    fused_pyr = [
        sum(gau_weights[k][lvl][..., None] * lap_inputs[k][lvl]
            for k in range(len(inputs)))
        for lvl in range(levels)
    ]

    # Reconstruct from coarsest to finest
    out = fused_pyr[-1]
    for lvl in range(levels - 2, -1, -1):
        h_lvl, w_lvl = fused_pyr[lvl].shape[:2]
        out = cv2.pyrUp(out, dstsize=(w_lvl, h_lvl)) + fused_pyr[lvl]

    logger.debug("multiscale_fusion: levels=%d, inputs=%d", levels, len(inputs))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5 — Luminance-Preserving Adaptive Gamma Correction
# ---------------------------------------------------------------------------

def adaptive_gamma_correction(
    img: np.ndarray,
    lo: float = 0.7,
    hi: float = 1.3,
) -> tuple[np.ndarray, float]:
    """
    Apply scene-adaptive gamma correction to the L* channel (CIE Lab).

    **Why adaptive?**
    Deep-water scenes can be globally dark (low mean L*) or bright near the
    surface.  A fixed gamma over- or under-corrects depending on the scene.
    This function derives gamma from the mean L* value:
    ``gamma = log(0.5) / log(mean_L*)``, which maps a scene of mean
    brightness 0.5 to gamma=1 (no change) and brightens darker scenes
    while slightly darkening bright ones.  Gamma is clamped to ``[lo, hi]``
    to prevent extreme corrections.

    Parameters
    ----------
    img : np.ndarray
        Input RGB float32 image, ``[0, 1]``.
    lo : float
        Lower bound for the computed gamma (prevents over-brightening dark scenes).
    hi : float
        Upper bound for the computed gamma (prevents over-darkening bright scenes).

    Returns
    -------
    tuple[np.ndarray, float]
        ``(gamma_corrected_image, applied_gamma)`` — the image and the scalar gamma
        used (useful for logging and header metadata).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l_chan = np.clip(lab[..., 0] / 100.0, 1e-4, 1.0)

    m = float(np.clip(l_chan.mean(), 0.05, 0.95))
    gamma = float(np.clip(np.log(0.5) / np.log(m), lo, hi))

    lab[..., 0] = ((l_chan ** gamma) * 100.0).astype(np.float32)
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

    logger.debug("adaptive_gamma: mean_L=%.3f → gamma=%.3f", m, gamma)
    return np.clip(out, 0.0, 1.0).astype(np.float32), gamma


# ---------------------------------------------------------------------------
# Phase 2 Stub — PyTorch Deep-Learning Enhancement
# ---------------------------------------------------------------------------

def torch_enhance(bgr: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    [STUB] Apply a trained PyTorch underwater enhancement model.

    Replace the body of this function in Phase 2 with:

        1. Pre-process: BGR uint8 → RGB tensor float32, normalise, add batch dim.
        2. Forward pass::

               with torch.no_grad():
                   output_tensor = model(input_tensor)

        3. Post-process: squeeze batch dim, clamp to [0,1], scale to uint8, BGR.

    The function signature must remain stable so callers do not change.

    Parameters
    ----------
    bgr : np.ndarray
        Input image in OpenCV BGR uint8 format.

    Returns
    -------
    np.ndarray
        Enhanced image in OpenCV BGR uint8 format.
    """
    logger.info("[STUB] torch_enhance — returning image unchanged (Phase 2 pending).")
    # TODO: load model weights and run inference
    return bgr


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def enhance_underwater_image(raw_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Execute the full NETRA Phase-1 enhancement pipeline on an RGB float image.

    Parameters
    ----------
    raw_rgb : np.ndarray
        Input RGB float32 image, shape ``(H, W, 3)``, values in ``[0, 1]``.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(enhanced_rgb, steps_applied)``
        · ``enhanced_rgb`` — RGB float32, ``[0, 1]``
        · ``steps_applied`` — ordered list of step labels for observability
    """
    steps: list[str] = []

    # 1 — Guided colour balancing
    balanced = color_balance(raw_rgb)
    steps.append(f"color_balance(alpha={PARAMS['alpha']})")

    # 2 — MPR contrast boost (on L* channel)
    in1 = mpr(balanced)
    steps.append(
        f"mpr(c={PARAMS['c']},t={PARAMS['t']},sigma={PARAMS['sigma']})"
    )

    # 3 — Normalized unsharp masking (on L* channel)
    in2 = normalized_unsharp(balanced)
    steps.append(f"unsharp(sigma={PARAMS['sigma'] / 3.0:.1f})")

    # 4 — Compute & normalize per-input weight maps
    w1 = _compute_aggregated_weights(in1)
    w2 = _compute_aggregated_weights(in2)
    w1_norm, w2_norm = _normalize_weights([w1, w2])

    # 5 — Multiscale Laplacian–Gaussian pyramid fusion
    fused = multiscale_fusion([in1, in2], [w1_norm, w2_norm])
    steps.append("pyramid_fusion(laplacian+gaussian)")

    # 6 — Adaptive gamma correction
    enhanced, gamma = adaptive_gamma_correction(fused)
    steps.append(f"adaptive_gamma(gamma={gamma:.3f})")

    return enhanced, steps


# ---------------------------------------------------------------------------
# Public API — entry point for FastAPI endpoint
# ---------------------------------------------------------------------------

def run_pipeline(raw_bytes: bytes) -> EnhancementResult:
    """
    Execute the full NETRA enhancement pipeline on raw multipart image bytes.

    This is the **single entry point** called by the FastAPI endpoint in
    ``main.py``.  It handles:

    1. Decoding raw bytes → BGR NumPy array (OpenCV).
    2. Colour space conversion: BGR → RGB float32 ``[0, 1]``.
    3. Running :func:`enhance_underwater_image`.
    4. Converting result back: RGB float32 → BGR uint8.
    5. Returning an :class:`EnhancementResult` with the processed image and
       a step log for the ``X-NETRA-Steps`` response header.

    Parameters
    ----------
    raw_bytes : bytes
        Raw image file bytes from the multipart upload.

    Returns
    -------
    EnhancementResult
        Contains the processed BGR image and ordered step labels.

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as a valid image (corrupt / unsupported).
    """
    steps_applied: list[str] = []

    # --- Decode raw bytes → BGR NumPy array ---
    np_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(
            "Failed to decode the uploaded file as an image. "
            "The file may be corrupt, truncated, or in an unsupported format."
        )

    h, w = bgr.shape[:2]
    logger.info("Decoded image: %d×%d px", w, h)
    steps_applied.append("decode")

    # --- Convert BGR uint8 → RGB float32 [0, 1] ---
    raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # --- Run physics-guided pipeline ---
    enhanced_rgb, pipeline_steps = enhance_underwater_image(raw_rgb)
    steps_applied.extend(pipeline_steps)

    # --- Convert RGB float32 → BGR uint8 ---
    enhanced_bgr = cv2.cvtColor(
        (enhanced_rgb * 255.0).clip(0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    logger.info("Pipeline complete — %d steps: %s", len(steps_applied), steps_applied)
    return EnhancementResult(image_bgr=enhanced_bgr, steps_applied=steps_applied)
