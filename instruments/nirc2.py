"""Everything NIRC2/Keck-specific: detector constants, the bad pixel mask,
frame classification from headers, and the north angle calculation.

Translated from AIR.jl's NIRC2.jl, constants.jl, angles.jl, and
generic_reduce/01_sort_frames.jl. The generic reduction code in
``reduction/`` takes these values as parameters, so supporting another
instrument means writing a module like this one.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date as _date

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

PLATE_SCALE = 0.009942  # arcsec / pixel, narrow camera

# Keck observatory
OBSERVATORY_LAT = +19.82525
OBSERVATORY_LON = -155.468889

# narrowband filters with no flats of their own -> acceptable substitutes
# (passed to reduction.calibrate.find_closest_flat / reduce_frame)
FLAT_EXCEPTIONS = {"NB2.108": ("K + clear", "Ks + clear", "Kp + clear")}

REQUIRED_HEADER_KEYWORDS = [
    "FILENAME", "FILTER", "ITIME", "COADDS", "NAXIS1", "NAXIS2",
    "SAMPMODE", "READS", "EL", "WCDMSTAT", "WCDTSTAT", "OBJECT", "SHRNAME",
]

_DEFAULT_BAD_PIXEL_MASK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "masks",
    "bad_pixel_mask_20230101.fits")

# the detector was replaced in late 2023, changing gain and well depth
_DETECTOR_SWAP_DATE = _date(2023, 11, 20)

# The polarimetry header keywords (PCUPR / PCUNAME, the HWP position) were
# only added to the NIRC2 headers around 2025-12-01. Earlier polarimetry
# data records the HWP angle in the OBJECT string instead, e.g.
# "h_hwp_modulation_hwp_40.0" or "pol_cal_imr_hwp_h_imr_0.0_hwp_90.0".
# The pipeline reads those as a fallback but warns, because other
# polarimetric keywords may be missing too.
POL_HEADER_EPOCH = _date(2025, 12, 1)

_HWP_IN_OBJECT = re.compile(r"hwp[_-](-?[\d.]+)\s*$", re.IGNORECASE)


def predates_pol_headers(header):
    """True if a frame was taken before the polarimetry header keywords
    existed (see :data:`POL_HEADER_EPOCH`)."""
    date_obs = header.get("DATE-OBS")
    if not date_obs:
        return False
    try:
        return _parse_date(date_obs) < POL_HEADER_EPOCH
    except Exception:
        return False


def check_pol_headers(header, strict=False):
    """Check that a frame carries the polarimetry header keywords.

    Data taken before ~2025-12-01 predates those keywords and needs special
    handling: the HWP angle has to be recovered from the OBJECT string, and
    other polarimetric metadata may simply be absent. Raises
    ``ValueError`` when ``strict``; otherwise warns and returns False.
    """
    if header.get("PCUPR") is not None:
        return True

    date_obs = header.get("DATE-OBS", "unknown")
    early = predates_pol_headers(header)
    msg = (f"Frame {header.get('FILENAME', '?')} ({date_obs}) has no PCUPR "
           f"keyword")
    if early:
        msg += (f"; data before {POL_HEADER_EPOCH} predates the NIRC2 "
                f"polarimetry header keywords and needs special handling "
                f"(HWP angle is read from OBJECT instead)")
    if strict:
        raise ValueError(msg)
    log.warning("%s", msg)
    return False


def _parse_date(date_obs):
    """Parse a DATE-OBS string like '2025-12-04'."""
    return _date.fromisoformat(str(date_obs)[:10])


def get_gain(date_obs):
    """Detector gain in photoelectrons per ADU."""
    return 4.0 if _parse_date(date_obs) < _DETECTOR_SWAP_DATE else 8.0


def get_readnoise(sampmode):
    """Read noise in photoelectrons, by sampling mode."""
    if sampmode == 2:
        return 50.0
    if sampmode == 3:
        return 15.0
    return 0.0


def get_saturation_limit(date_obs):
    """Saturation / linearity limit in ADU."""
    return 9000.0 if _parse_date(date_obs) < _DETECTOR_SWAP_DATE else 4500.0


def load_bad_pixel_mask(path=_DEFAULT_BAD_PIXEL_MASK):
    """Load the static NIRC2 bad pixel mask as a boolean array."""
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=bool)


#
# frame classification, from generic_reduce/01_sort_frames.jl
#

FLAT_ELEVATION = 45.0  # deg; dome flats are always taken at this elevation


def _small_angle_distance(a, b):
    """Angular distance between two (ra, dec) pairs in degrees."""
    (ra_a, dec_a), (ra_b, dec_b) = a, b
    return np.sqrt(((ra_a - ra_b) * np.cos(np.deg2rad(dec_a))) ** 2
                   + (dec_a - dec_b) ** 2)


def _at_flat_position(frame, arcsec_threshold):
    distance_arcsec = 3600.0 * _small_angle_distance(
        (0.0, FLAT_ELEVATION), (0.0, frame["EL"]))
    dm_open = str(frame["WCDMSTAT"]).lower() in ("open", "idle")
    dt_open = str(frame["WCDTSTAT"]).lower() in ("open", "idle")
    shutter_open = str(frame["SHRNAME"]).lower() == "open"
    return (distance_arcsec < arcsec_threshold and dm_open and dt_open
            and shutter_open)


def is_lampon_frame(frame, arcsec_threshold=100.0, lampoff_threshold=100.0):
    """Dome flat with the lamp on: telescope at the flat position, AO loops
    open, shutter open, and counts above ``lampoff_threshold``."""
    return (_at_flat_position(frame, arcsec_threshold)
            and np.median(frame.data) > lampoff_threshold)


def is_lampoff_frame(frame, arcsec_threshold=100.0, lampoff_threshold=100.0):
    """Dome flat with the lamp off: same as lamp-on but with low counts."""
    return (_at_flat_position(frame, arcsec_threshold)
            and np.median(frame.data) <= lampoff_threshold)


def is_sky_twilight_frame(frame):
    """Sky or twilight flat, identified from the OBJECT name."""
    obj = str(frame["OBJECT"]).lower()
    return "sky" in obj or "twi" in obj


def is_dark_frame(frame):
    """Dark: shutter closed."""
    return str(frame["SHRNAME"]).lower() == "closed"


def sort_frames(filenames, lampoff_threshold=100.0, arcsec_threshold=100.0):
    """Classify raw NIRC2 FITS files by type using their headers.

    Returns a dict of filename lists with keys ``"sci"``, ``"flats"``,
    ``"flats_sky"``, ``"flats_lampon"``, ``"flats_lampoff"``, ``"darks"``.
    If no lamp-off flats are found, lamp-on flats are treated as regular
    flats (matching AIR.jl). Frames missing required header keywords are
    dropped with a warning.
    """
    from utils.frame import Frame

    frames, kept_filenames = [], []
    for fn in filenames:
        frame = Frame.load(fn)
        missing = [k for k in REQUIRED_HEADER_KEYWORDS if k not in frame]
        if missing:
            log.warning("Frame %s is missing required header keywords %s, "
                        "removing from list...", fn, missing)
            continue
        frames.append(frame)
        kept_filenames.append(fn)

    sorted_files = {"sci": [], "flats": [], "flats_sky": [],
                    "flats_lampon": [], "flats_lampoff": [], "darks": []}

    for fn, frame in zip(kept_filenames, frames):
        if is_lampon_frame(frame, arcsec_threshold, lampoff_threshold):
            sorted_files["flats_lampon"].append(fn)
        elif is_lampoff_frame(frame, arcsec_threshold, lampoff_threshold):
            sorted_files["flats_lampoff"].append(fn)
        elif is_sky_twilight_frame(frame):
            sorted_files["flats_sky"].append(fn)
        elif is_dark_frame(frame):
            sorted_files["darks"].append(fn)
        else:
            sorted_files["sci"].append(fn)

    if not sorted_files["flats_lampoff"]:
        log.info("No lamp-off flats found, assuming regular flats...")
        sorted_files["flats"] = sorted_files["flats_lampon"]
        sorted_files["flats_lampon"] = []

    for kind, files in sorted_files.items():
        log.info("Found %d %s frames", len(files), kind)

    return sorted_files


#
# north angle, from angles.jl (originally adapted from pyKLIP)
#

ZP_OFFSET = -0.262  # deg, narrow camera zero point from Service et al. 2016


def _ten(value):
    """Sexagesimal string ('HH:MM:SS.S' or '+DD:MM:SS') to decimal float."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    sign = -1.0 if parts[0].strip().startswith("-") else 1.0
    numbers = [abs(float(p)) for p in parts]
    return sign * sum(n / 60.0**i for i, n in enumerate(numbers))


def par_angle(hour_angle, dec, lat):
    """Parallactic angle [deg] from hour angle [hours], declination [deg],
    and latitude [deg]. Source: pyKLIP."""
    ha_rad = np.deg2rad(hour_angle * 15.0)
    dec_rad = np.deg2rad(dec)
    lat_rad = np.deg2rad(lat)

    parallang = -np.arctan2(
        -np.sin(ha_rad),
        np.cos(dec_rad) * np.tan(lat_rad) - np.sin(dec_rad) * np.cos(ha_rad),
    )
    return np.rad2deg(parallang)


def calculate_north_angle(header):
    """Angle to north [deg] for a NIRC2 narrow-camera frame, including the
    smearing of the parallactic angle over the exposure.

    ``header`` is anything with dict-style access to the FITS keywords
    (a ``Frame`` works). Returns ``(mean_angle, angles_per_second)`` where
    the second element traces the vertical position angle through the
    exposure (a single-element list when there is no smear).

    Derotate an image by rotating it by ``-mean_angle``.
    """
    rotator_mode = header["ROTMODE"]
    rotator_position = header["ROTPOSN"]  # deg
    instrument_angle = header["INSTANGL"]  # deg

    if rotator_mode == "vertical angle":
        if "PARANTEL" in header:
            parang = header["PARANTEL"]
        else:
            parang = header["PARANG"]
        pa_deg = parang + rotator_position - instrument_angle + ZP_OFFSET
    elif rotator_mode == "position angle":
        # no smear correction needed in position angle mode
        pa_deg = rotator_position - instrument_angle + ZP_OFFSET
        return pa_deg, [pa_deg]
    elif rotator_mode == "stationary":
        return np.nan, [np.nan]
    else:
        raise ValueError(f"Unknown rotator mode {rotator_mode}")

    # estimate the parallactic angle smear over the exposure
    itime = header["ITIME"]
    coadds = header["COADDS"]
    sampmode = header["SAMPMODE"]
    multisam = header["MULTISAM"]
    naxis1 = header["NAXIS1"]
    dec = _ten(header["DEC"]) + header["DECOFF"]

    if "TOTEXP" in header:
        totexp = header["TOTEXP"]
    elif sampmode == 2:
        totexp = (itime + 0.18 * (naxis1 / 1024.0) ** 2) * coadds
    elif sampmode == 3:
        totexp = (itime + (multisam - 1) * 0.18 * (naxis1 / 1024.0) ** 2) * coadds
    else:
        raise ValueError(f"Cannot compute TOTEXP for SAMPMODE {sampmode}")

    totexp_hours = totexp / 3600.0

    # hour angle at the start of the exposure
    ha_hours = 24.0 * _ten(header["HA"]) / 360.0

    if totexp <= 1.0:  # under a second: no appreciable smear
        return pa_deg, [pa_deg]

    # vertical position angle at each second of the exposure
    n_seconds = int(3600 * totexp_hours)
    seconds = np.arange(n_seconds)
    ha_steps = ha_hours + (seconds + 1.0 + 0.001) / 3600.0
    vp = par_angle(ha_steps, dec, OBSERVATORY_LAT)
    vpref = vp[0]

    # handle PA wrapping across 0 <-> 360
    vp = np.where(vp < 0, vp + 360.0, vp)
    if vpref < 0:
        vpref += 360.0
    if vpref > 360:
        vpref -= 360.0
    if np.any(vp > 350) and np.any(vp < 10):
        vp = np.where(vp > 350, vp - 360.0, vp)

    vpmean = np.mean(vp[np.isfinite(vp)])
    if vpmean < 60 and vpref > 350:
        vpmean += 360.0
        vp = vp + 360.0

    mean_angle = pa_deg + (vpmean - vpref)
    return mean_angle, list(vp)


def make_frametable(frames, table_filename, fields=None):
    """Write a fixed-width text table of frame header values for a quick
    overview of a night's data."""
    field_widths = {
        "FILENAME": 18, "OBJECT": 16, "TARGNAME": 16, "RA": 12, "DEC": 12,
        "CAMNAME": 8, "DATE-OBS": 10, "UTC": 11, "ITIME": 8, "COADDS": 8,
        "FILTER": 20, "FWONAME": 10, "FWINAME": 10, "GRSNAME": 8, "EL": 6,
        "AZ": 6, "PARANG": 6, "ROTMODE": 16, "INSTANG": 8, "PARANTEL": 8,
        "SLITNAME": 10, "NAXIS1": 6, "NAXIS2": 6,
    }

    if fields is None:
        fields = list(field_widths)
    printed = {f: field_widths[f] for f in fields if f in field_widths}

    log.info("Writing frames table to %s", table_filename)
    with open(table_filename, "w") as io:
        io.write(" | ".join(f"{f:<{w}}" for f, w in printed.items()) + "\n")
        for frame in frames:
            row = " | ".join(f"{str(frame.get(f, '')):<{w}}"
                             for f, w in printed.items())
            io.write(row + "\n")


#
# polarimetry-mode instrument interface
#

from .base import PolarimetryData  # noqa: E402


class NIRC2PolarimetryData(PolarimetryData):
    """NIRC2-Pol implementation of the instrument interface.

    The Wollaston prism puts the two orthogonally-polarized beams on the
    top and bottom halves of the detector; the HWP (held in the PCU, header
    keyword PCUPR) modulates the signal through cycles of the four critical
    angles (0, 45, 22.5, 67.5 deg).
    """

    name = "NIRC2"
    plate_scale = PLATE_SCALE
    flat_exceptions = FLAT_EXCEPTIONS

    # HWP angle lives in PCUPR (the PCU rotation stage holding the HWP;
    # PCUNAME gives the named PCU position)
    modulator_keyword = "PCUPR"
    modulator_cycle_length = 4
    critical_angles = (0.0, 45.0, 22.5, 67.5)

    # image rotator keyword used in the polarimetric rotation model; note
    # ROTPDEST is 2x the physical rotator-to-bench angle (OBRT), so it
    # enters the model with a factor 2, not 4
    rotator_keyword = "ROTPDEST"

    # beam extraction geometry (detector rows/columns); tune per epoch by
    # checking the beam alignment on a bright star
    beam_height = 450       # rows in each beam cutout
    bottom_row_start = 0    # bottom beam: rows [0, beam_height)
    top_row_start = 508     # top beam: rows [start, start + beam_height)
    beam_x_offset = 13      # horizontal shift of top beam relative to bottom

    # HWP fast axis offset theta_off [deg] entering the rotation model;
    # set from the fast axis calibration log (load_fast_axis_offset)
    fast_axis_offset = 0.0

    def gain(self, header):
        return get_gain(header["DATE-OBS"])

    def saturation_limit(self, header):
        return get_saturation_limit(header["DATE-OBS"])

    def readnoise(self, header):
        return get_readnoise(header["SAMPMODE"])

    def bad_pixel_mask(self):
        return load_bad_pixel_mask()

    def sort_frames(self, filenames, **kwargs):
        return sort_frames(filenames, **kwargs)

    def north_angle(self, header):
        return calculate_north_angle(header)[0]

    def modulator_angle(self, header):
        """HWP angle [deg] from PCUPR, falling back to the OBJECT string for
        data predating the polarimetry header keywords."""
        angle = header.get(self.modulator_keyword)
        if angle is not None:
            return float(angle)

        check_pol_headers(header, strict=False)
        match = _HWP_IN_OBJECT.search(str(header.get("OBJECT", "")))
        if match:
            return float(match.group(1))
        raise ValueError(
            f"Cannot determine the HWP angle for "
            f"{header.get('FILENAME', 'frame')}: no {self.modulator_keyword} "
            f"keyword and no 'hwp_<angle>' in OBJECT "
            f"({header.get('OBJECT')!r}). Data before {POL_HEADER_EPOCH} "
            f"needs special handling.")

    def split_beams(self, frame):
        """Cut out the ordinary and extraordinary beams and register them:
        returns a ``(2, beam_height, nx - beam_x_offset)`` array with
        beam 0 = bottom, beam 1 = top, the top beam shifted by
        ``beam_x_offset`` columns so both cover the same sky."""
        data = frame.data if hasattr(frame, "data") else np.asarray(frame)

        bottom = data[self.bottom_row_start:
                      self.bottom_row_start + self.beam_height, :]
        top = data[self.top_row_start:
                   self.top_row_start + self.beam_height, :]

        stack = np.zeros((2, self.beam_height,
                          bottom.shape[1] - self.beam_x_offset))
        stack[0] = bottom[:, :-self.beam_x_offset]
        stack[1] = top[:, self.beam_x_offset:]
        return stack

    def qu_rotation_angle(self, header, fast_axis_offset=None):
        """Overall polarimetric rotation [deg] (SPIE Eq. 3)::

            theta_rot = -2*PARANG + 2*EL + 2*ROTPDEST + 4*theta_off

        assuming an idealized system; the full Mueller matrix model will
        replace this once calibrated (see polarimetry/mueller.py).
        """
        if fast_axis_offset is None:
            fast_axis_offset = self.fast_axis_offset

        parang = header["PARANG"]
        if parang < 0:
            parang += 360.0
        return (-2.0 * parang + 2.0 * header["EL"]
                + 2.0 * header[self.rotator_keyword]
                + 4.0 * fast_axis_offset)


#
# fast axis calibration
#
# Each time the instrument team runs a fast axis calibration (HWP rotated 0
# to 180 deg in 10 deg steps on a calibration source), the resulting offset
# is appended to a log file; reductions pull the most recent value on or
# before the observation date, or the user can override.
#

_DEFAULT_FAST_AXIS_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fast_axis_log.csv")


# Recommended background treatment per band (see PolarimetryData).
RECOMMENDED_BACKGROUND = {
    "Lp": ("dither", "mean_box"), "L": ("dither", "mean_box"),
    "Ms": ("dither", "mean_box"), "M": ("dither", "mean_box"),
    "J": ("annulus", "mean_box"), "H": ("annulus", "mean_box"),
    "K": ("annulus", "mean_box"), "Kp": ("annulus", "mean_box"),
    "Ks": ("annulus", "mean_box"),
}


def check_background_choice(band, method):
    """Warn if a background method is a poor fit for the observing band.

    L' and M sit on a large, structured thermal pedestal, so they need
    dither-pair subtraction or at least a mean box; in JHK an annulus
    around the source is usually cleanest.
    """
    rec = RECOMMENDED_BACKGROUND.get(str(band).strip())
    if rec is None or method is None:
        return True
    if method not in rec:
        log.warning("Background method %r is not recommended for %s band "
                    "(use one of %s)", method, band, " or ".join(rec))
        return False
    return True


def band_of(header):
    """Observing band of a frame, as used to key fast axis calibrations.

    Uses FWINAME (the wavelength filter wheel), e.g. "H", "Kp", "Lp",
    "H2O_ice"; falls back to the first token of FILTER.
    """
    band = header.get("FWINAME")
    if band:
        return str(band).strip()
    return str(header.get("FILTER", "")).split("+")[0].strip()


def load_fast_axis_offset(date_obs, band=None, log_file=_DEFAULT_FAST_AXIS_LOG,
                          strict_band=True):
    """Most recent fast axis offset [deg] on or before ``date_obs``, for the
    matching ``band``, from the calibration log.

    The offset depends on the observing band, so a calibration is only
    valid for data taken in the same filter: pass the band (see
    :func:`band_of`) and the log is filtered to it. With
    ``strict_band=False`` the search falls back to any band when no
    matching calibration exists, which is a guess — the returned value is
    then flagged in the log message.

    CSV columns: date, band, theta_off_deg, notes.
    """
    import csv

    obs = _parse_date(date_obs)
    entries = []
    with open(log_file) as f:
        # allow '#' comment lines anywhere, including above the header row
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if row.get("date"):
            entries.append((_parse_date(row["date"]),
                            str(row.get("band", "")).strip(),
                            float(row["theta_off_deg"])))

    usable = [e for e in sorted(entries) if e[0] <= obs]
    if band is not None:
        matching = [e for e in usable if e[1].lower() == str(band).lower()]
    else:
        matching = usable

    if not matching:
        if band is not None and not strict_band and usable:
            cal_date, cal_band, theta_off = usable[-1]
            log.warning("No %s fast axis calibration on or before %s; falling "
                        "back to the %s calibration from %s (%.4f deg) -- "
                        "the offset is band dependent, so treat this as "
                        "provisional", band, date_obs, cal_band, cal_date,
                        theta_off)
            return theta_off
        raise ValueError(
            f"No fast axis calibration for band {band!r} on or before "
            f"{date_obs} in {log_file}")

    cal_date, cal_band, theta_off = matching[-1]
    log.info("Using %s-band fast axis offset %.4f deg from the %s calibration",
             cal_band, theta_off, cal_date)
    return theta_off


def record_fast_axis_offset(date, band, theta_off, notes="",
                            log_file=_DEFAULT_FAST_AXIS_LOG):
    """Append a fast axis calibration result to the log file."""
    import csv

    new_file = not os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["date", "band", "theta_off_deg", "notes"])
        writer.writerow([str(date), str(band), f"{theta_off:.4f}", notes])


def fit_fast_axis_sequence(hwp_angles, fluxes):
    """Fit a fast axis calibration sequence: single-difference flux of a
    polarized calibration source vs. HWP angle (0 to 180 deg in 10 deg
    steps).

    The modulation follows ``A * cos(4 * (theta - theta_off)) + C``; the
    fitted phase gives the fast axis offset theta_off [deg], wrapped into
    (-22.5, 22.5] (the model is degenerate modulo 45 deg).
    """
    from scipy.optimize import curve_fit

    hwp_angles = np.asarray(hwp_angles, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)

    def model(theta, amp, theta_off, const):
        return amp * np.cos(np.radians(4.0 * (theta - theta_off))) + const

    amp0 = (np.nanmax(fluxes) - np.nanmin(fluxes)) / 2.0
    p0 = [amp0, 0.0, float(np.nanmean(fluxes))]
    params, _ = curve_fit(model, hwp_angles, fluxes, p0=p0)

    amp, theta_off, _ = params
    if amp < 0:  # fold negative amplitude into the phase
        theta_off += 22.5
    theta_off = (theta_off + 22.5) % 45.0 - 22.5

    log.info("Fast axis sequence fit: theta_off = %.4f deg", theta_off)
    return float(theta_off)
