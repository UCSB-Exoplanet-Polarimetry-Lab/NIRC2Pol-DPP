"""Everything NIRC2/Keck-specific: detector constants, the bad pixel mask,
frame classification from headers, and the north angle calculation.

Partially translated from AIR.jl's NIRC2.jl, constants.jl, angles.jl, and
generic_reduce/01_sort_frames.jl. The generic reduction code in
``reduction/`` takes these values as parameters, so supporting another
instrument means writing a module like this one.
"""

from __future__ import annotations

import logging
import os
from instruments.base import read_config
from utils.angles import (par_angle, sexagesimal_to_degrees,
                          small_angle_distance)
from utils.frame import Frame, parse_date_obs

import numpy as np

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nirc2.toml")

_CONFIG = read_config(CONFIG_PATH)

PLATE_SCALE = _CONFIG["instrument"]["plate_scale"]

# Keck observatory
OBSERVATORY_LAT = _CONFIG["observatory"]["latitude"]
OBSERVATORY_LON = _CONFIG["observatory"]["longitude"]

# Which flat type each band requires, and the fallback for unlisted bands.
DEFAULT_REQUIRED_FLAT_TYPE = _CONFIG["flat_type"]["default_flat_type"].strip().upper()
REQUIRED_FLAT_TYPE_BY_BAND = {
    band: value.strip().upper()
    for band, value in _CONFIG["flat_type"].items()
    if band != "default_flat_type"}

REQUIRED_HEADER_KEYWORDS = [
    "FILENAME", "FILTER", "ITIME", "COADDS", "NAXIS1", "NAXIS2",
    "SAMPMODE", "READS", "EL", "WCDMSTAT", "WCDTSTAT", "OBJECT", "SHRNAME",
]

_DEFAULT_BAD_PIXEL_MASK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "masks",
    "bad_pixel_mask_20230101.fits")

# the detector was replaced in late 2023, changing gain. 
# no data for pol mode should be from before this date, but we have the safeguard there just in case someone wants to use the basic
# functions in the DPP to analyze older data.
_DETECTOR_SWAP_DATE = _CONFIG["detector"]["swap_date"]

# The polarimetry header keywords (PCUPR / PCUNAME, the HWP position) were
# only added to the NIRC2 headers around 2025-12-01. Earlier polarimetry
# data records the HWP angle in the OBJECT string instead, e.g.
# "h_hwp_modulation_hwp_40.0" or "pol_cal_imr_hwp_h_imr_0.0_hwp_90.0".
# The pipeline reads those as a fallback but warns, because other
# polarimetric keywords may be missing too.
POL_HEADER_EPOCH = _CONFIG["polarimetry"]["pol_header_epoch"]

def predates_pol_headers(header):
    """True if a frame was taken before the polarimetry header keywords
    existed (see :data:`POL_HEADER_EPOCH`)."""
    date_obs = header.get("DATE-OBS")
    if not date_obs:
        return False
    try:
        return parse_date_obs(date_obs) < POL_HEADER_EPOCH
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


def get_gain(date_obs):
    """Detector gain in photoelectrons per ADU, either side of the swap."""
    before = parse_date_obs(date_obs) < _DETECTOR_SWAP_DATE
    return _CONFIG["detector"]["gain_before" if before else "gain_after"]


def get_readnoise(sampmode):
    """Read noise in photoelectrons, by sampling mode."""
    detector = _CONFIG["detector"]
    option = f"readnoise_sampmode_{sampmode}"
    if option not in detector:
        option = "readnoise_default"
    return detector[option]


def get_saturation_limit(date_obs):
    """Saturation / linearity limit in ADU, either side of the swap."""
    before = parse_date_obs(date_obs) < _DETECTOR_SWAP_DATE
    return _CONFIG["detector"][
        "saturation_before" if before else "saturation_after"]


def load_bad_pixel_mask(path=_DEFAULT_BAD_PIXEL_MASK):
    """Load the static NIRC2 bad pixel mask as a boolean array.

    Parameters
    ----------
    path : str, optional
        FITS file to read. Defaults to the mask shipped with the package.

    Returns
    -------
    ndarray of bool
        The mask, anything non-zero being bad.

    Notes
    -----
    Reading is :meth:`utils.Frame.load`, which already handles gzip and the
    primary HDU; the only NIRC2-specific part is which file, so there is no
    second loader to keep in step with the first.
    """
    return Frame.load(path).data.astype(bool)


#
# frame classification, from generic_reduce/01_sort_frames.jl
#

# deg; dome flats are always taken at this elevation
FLAT_ELEVATION = _CONFIG["instrument"]["flat_elevation"]


def _at_flat_position(frame, arcsec_threshold):
    """Is the telescope parked for dome flats?

    Parameters
    ----------
    frame : Frame
        Frame to test.
    arcsec_threshold : float
        Tolerance on the elevation, in arcseconds.

    Returns
    -------
    bool
        True when the elevation is at the flat-field position, both AO loops
        are open or idle, and the shutter is open.
    """
    distance_arcsec = 3600.0 * small_angle_distance(
        (0.0, FLAT_ELEVATION), (0.0, frame["EL"]))
    dm_open = str(frame["WCDMSTAT"]).lower() in ("open", "idle")
    dt_open = str(frame["WCDTSTAT"]).lower() in ("open", "idle")
    shutter_open = str(frame["SHRNAME"]).lower() == "open"
    return (distance_arcsec < arcsec_threshold and dm_open and dt_open
            and shutter_open)


def is_lampon_frame(frame, arcsec_threshold=100.0, min_flat_counts=100.0):
    """Dome flat with the lamp on: telescope at the flat position, AO loops
        open, shutter open, and counts above ``min_flat_counts``. Frames
        passing this test are dome flats (``FLATTYPE = "DOME"``); the lamp is
        how they are recognised, not a separate kind of flat.

        Lamp-*off* frames are not classified at all. They carry no useful
        information -- in JHK they are meaningless, and at L' the dome lamp is
        swamped by thermal background so sky flats are used regardless -- and
        the count threshold now only separates an illuminated flat from a dud.

    Parameters
    ----------
    frame : Frame
        Frame to test.
    arcsec_threshold : float, optional
        Tolerance on the flat-field elevation.
    min_flat_counts : float, optional
        Median counts below which the frame is not an illuminated flat.

    Returns
    -------
    bool
        True for a lamp-on dome flat.
    """
    return (_at_flat_position(frame, arcsec_threshold)
            and np.median(frame.data) > min_flat_counts)


def is_sky_twilight_frame(frame):
    """Sky or twilight flat, identified from the OBJECT name."""
    obj = str(frame["OBJECT"]).lower()
    return "sky" in obj or "twi" in obj


def is_dark_frame(frame):
    """Dark: shutter closed."""
    return str(frame["SHRNAME"]).lower() == "closed"


def sort_frames(filenames, min_flat_counts=100.0, arcsec_threshold=100.0):
    """Classify raw NIRC2 FITS files by type using their headers.

        Returns a dict of filename lists with keys ``"sci"``,
        ``"flats_dome"``, ``"flats_sky"``, ``"darks"``. Frames missing
        required header keywords are dropped with a warning.

        A flat is one of two kinds -- the dome screen or the twilight sky.
        Whether it is polarimetric is a separate property, decided later from
        the HWP angle, and either kind can be.

    Classify raw files by type, from their headers.

    Parameters
    ----------
    filenames : iterable of str
        Raw FITS paths.
    **kwargs
        Passed to :func:`sort_frames`.

    Returns
    -------
    dict
        Filename lists under ``"sci"``, ``"flats_dome"``, ``"flats_sky"``
        and ``"darks"``.
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

    sorted_files = {"sci": [], "flats_dome": [], "flats_sky": [],
                    "darks": []}

    for fn, frame in zip(kept_filenames, frames):
        if is_lampon_frame(frame, arcsec_threshold, min_flat_counts):
            sorted_files["flats_dome"].append(fn)
        elif is_sky_twilight_frame(frame):
            sorted_files["flats_sky"].append(fn)
        elif is_dark_frame(frame):
            sorted_files["darks"].append(fn)
        else:
            sorted_files["sci"].append(fn)

    for kind, files in sorted_files.items():
        log.info("Found %d %s frames", len(files), kind)

    return sorted_files


# deg, narrow camera zero point from Service et al. 2016
ZP_OFFSET = _CONFIG["instrument"]["zp_offset"]


def calculate_north_angle(header):
    """Angle to north [deg] for a NIRC2 narrow-camera frame, including the
    smearing of the parallactic angle over the exposure.

    ``header`` is anything with dict-style access to the FITS keywords
    (a ``Frame`` works). Returns ``(mean_angle, angles_per_second)`` where
    the second element traces the vertical position angle through the
    exposure (a single-element list when there is no smear).

    Derotate an image by rotating it by ``-mean_angle``.

    Adapted from angles.jl in AIR.jl and pyklip
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
    dec = sexagesimal_to_degrees(header["DEC"]) + header["DECOFF"]

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
    ha_hours = 24.0 * sexagesimal_to_degrees(header["HA"]) / 360.0

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
        overview of a night's data.

    Parameters
    ----------
    filenames : iterable of str
        Frames to tabulate.
    keywords : iterable of str, optional
        Header keywords to include as columns.
    sort_by : str, optional
        Column to sort on.

    Returns
    -------
    astropy.table.Table
        One row per frame -- a quick way to see what a night contains.
    """
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

    name = _CONFIG["instrument"]["name"]
    plate_scale = PLATE_SCALE
    required_flat_types = REQUIRED_FLAT_TYPE_BY_BAND
    default_required_flat_type = DEFAULT_REQUIRED_FLAT_TYPE

    # HWP angle lives in PCUPR (the PCU rotation stage holding the HWP;
    # PCUNAME gives the named PCU position)
    modulator_keyword = _CONFIG["polarimetry"]["modulator_keyword"]

    # PCUNAME names the PCU stage position: the HWP is in the beam at
    # hwp_center, and out of it parked at home or pointed at the telescope.
    modulator_name_keyword = "PCUNAME"
    modulator_in_names = ("hwp_center",)
    modulator_out_names = ("home", "telescope")
    modulator_cycle_length = _CONFIG["polarimetry"]["modulator_cycle_length"]
    critical_angles = tuple(_CONFIG["polarimetry"]["critical_angles"])

    # image rotator keyword used in the polarimetric rotation model; note
    # ROTPDEST is 2x the physical rotator-to-bench angle (OBRT), so it
    # enters the model with a factor 2, not 4
    rotator_keyword = _CONFIG["polarimetry"]["rotator_keyword"]

    # Beam extraction geometry (detector rows/columns). Measured from the
    # data by :func:`reduction.fit_beam_geometry` and assigned before
    # anything splits the beams.
    _announced_beam_geometry = False

    # Where reduction.fit_beam_geometry starts looking. Not a value to
    # reduce with: the geometry is measured from the data every time.
    beam_geometry_search = (_CONFIG["beam_geometry"]["search_top_row_start"],
                            _CONFIG["beam_geometry"]["search_beam_x_offset"])

    def beam_geometry_seed(self):
        """Starting point for the beam geometry search, as ``(top, xoff)``."""
        return self.beam_geometry_search

    beam_height = 450       # rows in each beam cutout
    bottom_row_start = 0    # bottom beam: rows [0, beam_height)
    top_row_start = None    # top beam: rows [start, start + beam_height)
    beam_x_offset = None    # horizontal shift of top beam relative to bottom

    # HWP fast axis offset theta_off [deg] entering the rotation model.
    # There is no trusted automatic source for this yet: it must be determined
    # on sky and set explicitly. 
    fast_axis_offset = 0.0

    def gain(self, header):
        """Detector gain [e-/ADU] for this frame's epoch.

        Parameters
        ----------
        header : Frame or Header
            Frame whose ``DATE-OBS`` selects the epoch.

        Returns
        -------
        float
            The gain. NIRC2's detector was replaced in late 2023, changing both
            gain and well depth.
        """
        return get_gain(header["DATE-OBS"])

    def saturation_limit(self, header):
        """Saturation level [ADU] for this frame's epoch.

        Parameters
        ----------
        header : Frame or Header
            Frame whose ``DATE-OBS`` selects the epoch.

        Returns
        -------
        float
            Level above which pixels are treated as saturated and given the wider
            "+" shaped mask, since saturation bleeds along detector columns.
        """
        return get_saturation_limit(header["DATE-OBS"])

    def readnoise(self, header):
        """Read noise [e-] for this frame.

        Parameters
        ----------
        header : Frame or Header
            Frame to describe.

        Returns
        -------
        float
            Read noise in electrons.
        """
        return get_readnoise(header["SAMPMODE"])

    def bad_pixel_mask(self):
        """Static detector bad-pixel mask.

        Returns
        -------
        ndarray of bool
            True on known-bad pixels, loaded from ``instruments/masks/``. Dated
            2023, so it predates the current detector: defects that have grown
            since are only partly covered.
        """
        return load_bad_pixel_mask()

    def sort_frames(self, filenames, **kwargs):
        """Classify raw files by type, from their headers.

        Parameters
        ----------
        filenames : iterable of str
            Raw FITS paths.
        **kwargs
            Passed to :func:`sort_frames`.

        Returns
        -------
        dict
            Filename lists under ``"sci"``, ``"flats_dome"``,
            ``"flats_sky"`` and ``"darks"``.
        """
        return sort_frames(filenames, **kwargs)

    def north_angle(self, header):
        """Angle from image up to celestial north [deg].

        Parameters
        ----------
        header : Frame or Header
            Frame to describe.

        Returns
        -------
        float
            The mean angle; derotate by minus this. See
            :func:`calculate_north_angle` for the full model, including the
            parallactic smear through a long exposure.
        """
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

    def split_beams(self, frame, top_row_start=None, beam_x_offset=None):
        """Cut out the ordinary and extraordinary beams and register them.

        Parameters
        ----------
        frame : Frame or ndarray
            Full detector frame.
        top_row_start : int, optional
            First detector row of the top beam. Defaults to the instrument
            attribute; pass it explicitly to try a trial geometry, as
            :meth:`fit_beam_geometry` does.
        beam_x_offset : int, optional
            Column shift of the top beam relative to the bottom one.
            Defaults to the instrument attribute.

        Returns
        -------
        ndarray
            ``(2, beam_height, nx - beam_x_offset)``, beam 0 = bottom and
            beam 1 = top, the top beam shifted by ``beam_x_offset`` columns
            so both cover the same sky.

        Raises
        ------
        ValueError
            If the geometry is unset

        Notes
        -----
        The geometry has to be set before this is called. It is measured
        from the data by :func:`reduction.fit_beam_geometry`, which is a
        standard step of the reduction
        """
        top_row_start = (self.top_row_start if top_row_start is None
                         else top_row_start)
        beam_x_offset = (self.beam_x_offset if beam_x_offset is None
                         else beam_x_offset)
        if top_row_start is None or beam_x_offset is None:
            raise ValueError(
                f"{type(self).__name__} has no beam geometry "
                f"(top_row_start={top_row_start!r}, "
                f"beam_x_offset={beam_x_offset!r}). It is measured from the "
                "data rather than defaulted. Measure it with "
                "reduction.fit_beam_geometry(instrument, frames) and assign "
                "the result before reducing.")
        
        data = np.asarray(frame.data if hasattr(frame, "header") else frame)

        bottom = data[self.bottom_row_start:
                      self.bottom_row_start + self.beam_height, :]
        top = data[top_row_start:top_row_start + self.beam_height, :]

        stack = np.zeros((2, self.beam_height,
                          bottom.shape[1] - beam_x_offset))
        stack[0] = bottom[:, :-beam_x_offset]
        stack[1] = top[:, beam_x_offset:]
        return stack

    def check_background_choice(self, header):
        """Warn once if the background method suits the band badly.

        Parameters
        ----------
        header : Header
            A frame from the dataset, for reading its band.
        """
        if type(self)._warned_background_choice:
            return
        type(self)._warned_background_choice = True
        check_background_choice(band_of(header), self.background_method)

    def describe_beam_geometry(self):
        """Where the two beams were cut from, as ``(top_row_start, x_offset)``.

        Returns
        -------
        str
            The pair actually used. It is measured per reduction rather than
            recorded per epoch, so without this a product could not say
            which geometry produced it.
        """
        return f"({self.top_row_start}, {self.beam_x_offset})"

    def occulting_radius(self, header):
        """Occulting mask radius [px] from SLITNAME, or None if unocculted.

        NIRC2 names its coronagraphic spots by *diameter* in milliarcsec, so
        ``corona150`` is 150 mas across: 7.5 px radius at the narrow-camera
        plate scale, and ``corona400`` is 20.1 px.
        """
        name = str(header.get("SLITNAME", "")).strip().lower()
        if not name.startswith("corona"):
            return None
        try:
            diameter_mas = float(name[len("corona"):])
        except ValueError:
            log.warning("Unrecognized coronagraph name %r; treating as "
                        "unocculted", name)
            return None
        return 0.5 * diameter_mas / 1000.0 / PLATE_SCALE

    def qu_rotation_angle(self, header, fast_axis_offset=None):
        """Overall polarimetric rotation [deg] (SPIE Eq. 3)::

                    theta_rot = -2*PARANG + 2*EL + 2*ROTPDEST + 4*theta_off

                assuming an idealized system; the full Mueller matrix model will
                replace this once calibrated (see polarimetry/mueller.py).

        Parameters
        ----------
        header : Frame or Header
            Frame supplying PARANG, EL and the rotator position.
        fast_axis_offset : float, optional
            theta_off in degrees. When omitted the instrument's attribute is used,
            and a one-time warning fires if that is still the uncalibrated 0.

        Returns
        -------
        float
            theta_rot in degrees, by which Q/U rotate into the sky frame.
        """
        if fast_axis_offset is None:
            # Only warn when the caller did not say: passing 0.0 explicitly
            # is a legitimate request, and polarimetry.fast_axis does exactly
            # that to evaluate the rotation at zero offset before scanning.
            fast_axis_offset = self.fast_axis_offset
            if (fast_axis_offset == 0.0
                    and not type(self)._warned_uncalibrated_offset):
                type(self)._warned_uncalibrated_offset = True
                log.warning(
                    "Fast axis offset is still the uncalibrated default "
                    "(0 deg), so Q/U are not rotated into the sky frame "
                    "correctly. Determine theta_off on sky and pass it "
                    "explicitly.")

        parang = header["PARANG"]
        if parang < 0:
            parang += 360.0
        return (-2.0 * parang + 2.0 * header["EL"]
                + 2.0 * header[self.rotator_keyword]
                + 4.0 * fast_axis_offset)


# Recommended background treatment per band (see PolarimetryData).
RECOMMENDED_BACKGROUND = {band: tuple(methods)
                          for band, methods in _CONFIG["background"].items()}


def check_background_choice(band, method):
    """Warn if a background method is a poor fit for the observing band.

    Parameters
    ----------
    band : str
        Observing band, e.g. from :func:`band_of`.
    method : str or None
        Background method being used.

    Returns
    -------
    bool
        True if the choice is recommended for the band. Unknown bands and a
        None method pass, since there is nothing to advise.
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
    """Observing band of a frame.

    Uses FWINAME (the wavelength filter wheel), e.g. "H", "Kp", "Lp",
    "H2O_ice"; falls back to the first token of FILTER.
    """
    band = header.get("FWINAME")
    if band:
        return str(band).strip()
    return str(header.get("FILTER", "")).split("+")[0].strip()
