"""What is known about a polarized standard star, independently of us.

Everything here describes a *source*: a polarization someone else measured,
and the dust physics needed to carry it to the band we observed in. Nothing
here touches an instrument or a reduction. The fitting that uses it lives in
:mod:`nirc2pol.polarimetry.fast_axis`.

**There is no bundled catalogue, deliberately.** The caller supplies the
numbers, exactly as ``fast_axis_method="fixed"`` makes the caller supply
``theta_off``. A table of standards shipped inside the package is a table that
goes stale silently and gets cited by accident; a value written in the script
that used it can be traced to whoever chose it.

The one thing worth knowing before reading further: **the angle and the degree
are not equally trustworthy, and they fail independently.** For interstellar
polarization the position angle is set by the projected orientation of the
Galactic magnetic field and is very nearly wavelength-independent, so an
optical catalogue angle carries to L' essentially unchanged. The degree is the
entire content of the Serkowski law and does not carry at all -- 3.8 um is a
long way outside where that law is constrained. In the fit these land in
different places (the angle in a phase, the degree in a modulus), so a bad
degree costs you the polarimetric efficiency and leaves the fast axis offset
alone. See :func:`serkowski_p` and
:func:`nirc2pol.polarimetry.fast_axis.fit_theta_off_polstd`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolarizedStandard:
    """A star whose linear polarization on sky is known from elsewhere.

    Attributes
    ----------
    name : str
        Whatever you call it. Recorded in logs and provenance, not matched
        against anything.
    p : float
        Degree of linear polarization as a **fraction**, not a percent:
        0.0093 for 0.93%. In the band the data was taken in -- if the
        catalogue value is optical, put it through :func:`serkowski_p`
        first rather than passing it here directly.
    theta : float
        Position angle of the polarization on sky [deg], in the same
        convention the pipeline's AoLP uses: ``0.5*arctan2(U, Q)`` on the
        derotated sky-frame cube. Whether that is degrees east of north
        depends on the image parity, which is why a fit against this
        absorbs any constant north-angle or parity error -- see the warning
        in :func:`nirc2pol.polarimetry.fast_axis.fit_theta_off_polstd`.
    p_err, theta_err : float, optional
        Uncertainties, ``p_err`` as a fraction and ``theta_err`` in degrees.
        **Two separate fields on purpose.** They are not interchangeable and
        are usually wildly different in quality: a p extrapolated to L' may
        be worth no better than a factor of two while the angle is worth two
        degrees. ``theta_err`` is the one the fit weights by, since the angle
        is what constrains the offset.
    band : str, optional
        Which band ``p`` and ``theta`` describe, as ``FWINAME`` spells it.
        Not checked against the data -- it is here so a product's provenance
        records which value was used.
    reference : str, optional
        Where the numbers came from. Worth filling in: this is the one input
        to the fast axis offset that cannot be recomputed from the data.

    Notes
    -----
    A standard is only a standard for the geometry it was measured in. For an
    interstellar-polarization star the polarization is imposed by dust along
    the sightline and is stable, which is what makes these usable at all; for
    an intrinsically polarized source (a Be star, an evolved star with a
    variable envelope) it is not, and a catalogue value from another epoch is
    not a calibration.
    """

    name: str
    p: float
    theta: float
    p_err: float = None
    theta_err: float = None
    band: str = None
    reference: str = None

    @property
    def stokes(self):
        """The sky-frame normalized Stokes vector as one complex number.

        ``p * exp(2j * theta)`` with ``theta`` in radians -- the form the fit
        actually consumes, since ``q + i*u`` rotates as a plain complex
        number under :func:`nirc2pol.polarimetry.rotate_qu`.
        """
        return self.p * np.exp(2j * np.radians(self.theta))

    def describe(self):
        """One-line summary for logs and FITS provenance."""
        s = f"{self.name}: p={100 * self.p:.3f}%"
        if self.p_err is not None:
            s += f"+/-{100 * self.p_err:.3f}"
        s += f" at {self.theta:.2f}deg"
        if self.theta_err is not None:
            s += f"+/-{self.theta_err:.2f}"
        if self.band:
            s += f" ({self.band})"
        if self.reference:
            s += f" [{self.reference}]"
        return s


def serkowski_p(wavelength_um, p_max, lambda_max_um, K=None):
    """Interstellar polarization at one wavelength, from the Serkowski law.

    ``p(lambda) = p_max * exp(-K * ln^2(lambda_max / lambda))``

    Parameters
    ----------
    wavelength_um : float or array_like
        Where you want p, in microns. NIRC2 L' is 3.776, Kp 2.124.
    p_max : float
        Peak degree of polarization, as a fraction. Whatever units go in
        come out.
    lambda_max_um : float
        Wavelength of the peak, in microns. Typically 0.45-0.8 for Galactic
        sightlines, 0.55 being the usual default when it is unmeasured.
    K : float, optional
        Width of the curve. Defaults to Whittet's empirical
        ``K = 0.01 + 1.66 * lambda_max``, which ties the width to the peak
        and so needs no extra input.

    Returns
    -------
    float or ndarray
        The degree of polarization, in the units of ``p_max``.

    Warnings
    --------
    **This is an extrapolation at L', and a long one.** The Serkowski law is
    fitted on optical and near-IR data around a peak near 0.55 um; 3.8 um is
    several times lambda_max, out where the observed polarization is better
    described by a power law than by this curve and where the two forms
    disagree by more than either is determined. Treat the number as an order
    of magnitude with a large fractional uncertainty, put that uncertainty in
    ``PolarizedStandard.p_err``, and do not quote it as a measurement.

    The saving grace is that this is the input the fast axis offset barely
    depends on. ``p`` sets the modulus of the source term and therefore the
    fitted polarimetric efficiency; the *angle* sets the phase and therefore
    the offset. A p wrong by a factor of two moves the efficiency by a factor
    of two and leaves theta_off where it was.

    Notes
    -----
    The angle carries far better than the degree: the position angle of
    interstellar polarization traces the projected field orientation and is
    nearly wavelength-independent, so an optical ``theta`` is usable in the
    infrared as it stands. The exception is a sightline crossing two clouds
    whose fields are differently oriented, which rotates the angle slowly
    with wavelength -- real, usually small, and not something this function
    models.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    if np.any(wavelength_um <= 0):
        raise ValueError("wavelength must be positive")
    if lambda_max_um <= 0:
        raise ValueError("lambda_max_um must be positive")
    if K is None:
        K = 0.01 + 1.66 * lambda_max_um
    p = p_max * np.exp(-K * np.log(lambda_max_um / wavelength_um) ** 2)
    return float(p) if np.ndim(p) == 0 else p
