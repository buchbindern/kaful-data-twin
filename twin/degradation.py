"""
PowerLawWear (M5b) — the degradation model.

State-space, Paris-style power law:   dw/dn = a * w**p   (a>0, typically p>1)

This is a genuine Markov state model in wear: the wear rate depends only on the
current wear, not on the cut number. That is exactly the state-transition the
particle filter forward-propagates (M6) and Monte Carlo projects to the failure
threshold (M7). We advance/threshold in CLOSED FORM (exact integral), not Euler,
so there is no step-size error.

It models the WEAR-OUT regime only. Running-in (high rate at low wear) is not
represented here — by design; the twin's wide early-life uncertainty and the
observation updates carry the state until wear-out dynamics dominate.

Integral of dw/dn = a*w**p (for p != 1), advancing dn cuts from wear w:
    w(dn) = [ w**(1-p) + a*(1-p)*dn ] ** (1/(1-p))
For p>1 this has a finite-time singularity (w -> inf), which is the realistic
"runaway to failure" behavior; the threshold is crossed before the singularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class PowerLawWear:
    a: float
    p: float

    def rate(self, w):
        """Instantaneous wear rate dw/dn at wear w."""
        return self.a * np.power(w, self.p)

    def advance(self, w, dn: float = 1.0):
        """Wear after advancing `dn` cuts from wear `w` (closed form).
        Works on scalars or numpy arrays (for particle clouds). Past the
        singularity, returns +inf."""
        w = np.asarray(w, dtype=float)
        if abs(self.p - 1.0) < 1e-12:
            out = w * np.exp(self.a * dn)
        else:
            rhs = np.power(w, 1.0 - self.p) + self.a * (1.0 - self.p) * dn
            with np.errstate(invalid="ignore"):
                out = np.where(rhs > 0.0,
                               np.power(np.clip(rhs, 1e-300, None), 1.0 / (1.0 - self.p)),
                               np.inf)
        return float(out) if out.ndim == 0 else out

    def cuts_to_threshold(self, w, threshold: float):
        """Cuts remaining from wear `w` until reaching `threshold` (closed form).
        0 if already at/past threshold. Scalar or array in `w`."""
        w = np.asarray(w, dtype=float)
        if abs(self.p - 1.0) < 1e-12:
            n = np.log(threshold / w) / self.a
        else:
            n = (np.power(threshold, 1.0 - self.p) - np.power(w, 1.0 - self.p)) \
                / (self.a * (1.0 - self.p))
        n = np.maximum(n, 0.0)
        return float(n) if n.ndim == 0 else n

    # ---- fitting on a labeled reference run ----
    @staticmethod
    def detect_onset(cuts, wears) -> float:
        """Data-driven wear-out onset = the cut of minimum (smoothed) wear rate,
        after skipping the initial running-in transient."""
        cuts = np.asarray(cuts, float); wears = np.asarray(wears, float)
        rate = np.gradient(wears, cuts)
        k = 5
        srate = np.convolve(rate, np.ones(k) / k, mode="same")
        start = int(0.15 * len(cuts))
        i = start + int(np.argmin(srate[start:]))
        return float(cuts[i])

    @classmethod
    def fit(cls, cuts, wears, onset_cut: float | None = None):
        """Fit (a, p) on the wear-out region (cut >= onset). Returns (model, info)."""
        cuts = np.asarray(cuts, float); wears = np.asarray(wears, float)
        if onset_cut is None:
            onset_cut = cls.detect_onset(cuts, wears)
        mask = cuts >= onset_cut
        n0 = cuts[mask][0]; w0 = wears[mask][0]

        def integral(n, a, p):
            return np.power(np.power(w0, 1 - p) + a * (1 - p) * (n - n0), 1 / (1 - p))

        (a, p), _ = curve_fit(integral, cuts[mask], wears[mask],
                              p0=[1e-3, 1.5], maxfev=200000)
        model = cls(a=float(a), p=float(p))
        pred = integral(cuts[mask], a, p)
        rmse_um = float(np.sqrt(np.mean((pred - wears[mask]) ** 2)) * 1000)
        info = {"onset_cut": float(onset_cut), "w0_at_onset": float(w0),
                "n_fit_points": int(mask.sum()), "rmse_um": rmse_um}
        return model, info
