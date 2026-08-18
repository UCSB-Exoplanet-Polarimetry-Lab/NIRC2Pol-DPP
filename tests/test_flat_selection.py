"""Flat selection policy: which flat is allowed to calibrate which frame.

This is policy, not implementation detail — the rules were specified rather
than derived, and getting them wrong produces a reduction that looks fine.
"""

import logging

import numpy as np
import pytest

from instruments.nirc2 import NIRC2PolarimetryData
from reduction.calibrate import find_closest_flat, reduce_frame
from reduction.masters import (flat_sort_key, make_master_darks,
                               make_master_flats, required_flat_type_for)
from utils.frame import Frame
from utils.provenance import describe

# Which flat a band requires is a property of the instrument, not of the
# reduction code, so the table comes from the instrument here exactly as it
# does in a real reduction.
FLAT_TYPES = NIRC2PolarimetryData.required_flat_types
DEFAULT_FLAT_TYPE = NIRC2PolarimetryData.default_required_flat_type


def _flat(filtername="Kp + Wollaston", band="Kp", flattype="LAMP", n=64,
          nframes=3, polflat=False):
    """A master flat with the given filter, band and type.

    Returns
    -------
    Frame
        The synthetic flat, carrying the header keywords the matcher reads.
    """
    return Frame(np.ones((n, n)),
                 {"FILTER": filtername, "FWINAME": band, "NAXIS1": n,
                  "NAXIS2": n, "FLATTYPE": flattype, "NFRAMES": nframes,
                  "POLFLAT": polflat})


def _science(filtername="Kp + Wollaston", band="Kp", n=64):
    """A science frame needing a flat.

    Returns
    -------
    Frame
        The synthetic frame.
    """
    return Frame(np.ones((n, n)),
                 {"FILTER": filtername, "FWINAME": band, "NAXIS1": n,
                  "NAXIS2": n, "FILENAME": "sci.fits", "FRAMENO": 932,
                  "ITIME": 1.0, "COADDS": 1})


def _find_flat(*args, **kwargs):
    """``find_closest_flat`` with the instrument's flat-type table, the way a
    real reduction calls it."""
    kwargs.setdefault("required_flat_types", FLAT_TYPES)
    kwargs.setdefault("default_required_flat_type", DEFAULT_FLAT_TYPE)
    return find_closest_flat(*args, **kwargs)


def _sort_key(flat, required_type=None):
    """``flat_sort_key`` with the instrument's table."""
    return flat_sort_key(flat, required_type, FLAT_TYPES, DEFAULT_FLAT_TYPE)


# --- which type each band requires -------------------------------------

@pytest.mark.parametrize("band,expected", [
    ("Lp", "SKY"), ("L", "SKY"),
    ("J", "LAMP"), ("H", "LAMP"), ("K", "LAMP"), ("Kp", "LAMP"),
    ("Ks", "LAMP"),
])
def test_required_flat_type_by_band(band, expected):
    """Sky flats in the thermal infrared, where the dome lamp is swamped by
    thermal background; lamp flats in the near infrared."""
    assert required_flat_type_for(band, flat_types=FLAT_TYPES,
                                 default=DEFAULT_FLAT_TYPE) == expected


def test_required_flat_type_override_wins():
    """Some observers want skies in JHK too."""
    assert required_flat_type_for("Kp", override="SKY",
                                  flat_types=FLAT_TYPES) == "SKY"
    assert required_flat_type_for("Kp", override="sky",
                                  flat_types=FLAT_TYPES) == "SKY"


# --- the requirement is enforced, not merely preferred -----------------

def test_wrong_flat_type_raises():
    """An L' frame offered only a lamp flat refuses to reduce."""
    with pytest.raises(ValueError, match="requires a SKY flat"):
        _find_flat(_science("Lp + Wollaston", "Lp"),
                          [_flat("Lp + Wollaston", "Lp", "LAMP")])


def test_right_flat_type_is_accepted():
    """An L' frame takes a sky flat without complaint."""
    _, got = _find_flat(_science("Lp + Wollaston", "Lp"),
                               [_flat("Lp + Wollaston", "Lp", "SKY")])
    assert got is not None and got["FLATTYPE"] == "SKY"


def test_override_downgrades_the_error_and_is_recorded():
    """The override proceeds and records FLATMISM, so it stays auditable."""
    _, got = _find_flat(_science("Lp + Wollaston", "Lp"),
                               [_flat("Lp + Wollaston", "Lp", "LAMP")],
                               allow_flat_type_mismatch=True)
    assert got is not None
    assert got["FLATMISM"] is True, "the substitution must be recorded"


def test_explicit_required_type_overrides_the_band_rule():
    """An explicit required type wins over the band default."""
    _, got = _find_flat(_science("Lp + Wollaston", "Lp"),
                               [_flat("Lp + Wollaston", "Lp", "LAMP")],
                               required_flat_type="LAMP")
    assert got is not None


# --- the filter must match; size and exposure are not the same ---------

def test_filter_must_match():
    """A flat in the wrong filter is never used, whatever else matches. It
    now surfaces as a refusal rather than a quiet unflattened reduction."""
    with pytest.raises(ValueError, match="No usable flat"):
        _find_flat(_science("Kp + Wollaston", "Kp"),
                   [_flat("H + Wollaston", "H", "LAMP")])
    _, got = _find_flat(_science("Kp + Wollaston", "Kp"),
                        [_flat("H + Wollaston", "H", "LAMP")],
                        allow_no_flat=True)
    assert got is None, ("a flat in the wrong filter describes the wrong "
                         "throughput and must not be used")


def test_exposure_settings_are_ignored():
    """A flat is normalized, so ITIME and COADDS are irrelevant."""
    flat = _flat()
    flat["ITIME"], flat["COADDS"] = 30.0, 1
    sci = _science()
    sci["ITIME"], sci["COADDS"] = 0.45, 45
    _, got = _find_flat(sci, [flat])
    assert got is not None


def test_larger_flat_is_trimmed():
    """A full-frame flat is cropped to a subarray frame and marked FLATTRIM."""
    _, got = _find_flat(_science(n=32), [_flat(n=64)])
    assert got is not None
    assert got.shape == (32, 32)
    assert got["FLATTRIM"] is True


def test_smaller_flat_is_refused():
    """A flat smaller than the frame cannot cover it, so it is not matched --
    and with nothing else to fall back on, that is now a refusal rather than
    a quiet unflattened reduction."""
    with pytest.raises(ValueError, match="No usable flat"):
        _find_flat(_science(n=64), [_flat(n=32)])
    _, got = _find_flat(_science(n=64), [_flat(n=32)], allow_no_flat=True)
    assert got is None, "a flat smaller than the frame cannot calibrate it"


# --- ordering among valid flats ----------------------------------------

def test_flat_sort_order():
    """Polarimetric first, then dark-subtracted, then the required type,
    then whichever was built from the most frames."""
    pol = _flat(polflat=True, nframes=3)
    nodark = _flat(flattype="LAMP+NODARK", nframes=9)
    wrong_type = _flat(flattype="SKY", nframes=9)
    few = _flat(nframes=3)
    many = _flat(nframes=9)

    order = sorted([nodark, wrong_type, few, many, pol], key=_sort_key)
    assert order[0] is pol
    assert order[1] is many, "more frames wins among equals"
    assert order[2] is few
    assert order[3] is wrong_type, "the non-required real type ranks above "\
                                   "a darkless one"
    assert order[4] is nodark


# --- what happens when the instrument table is missing -----------------

def test_without_a_table_the_requirement_is_skipped_and_announced(caplog):
    """Which flat a band needs lives on the instrument, so generic code
    cannot check it alone. Skipping quietly is how a wrong flat slips
    through looking plausible, so it has to say so."""
    import reduction.calibrate as calibrate
    calibrate._WARNED_NO_FLAT_TYPE_TABLE = False
    with caplog.at_level(logging.WARNING):
        _, got = find_closest_flat(_science("Lp + Wollaston", "Lp"),
                                   [_flat("Lp + Wollaston", "Lp", "LAMP")])
    assert got is not None, "no table means no requirement, so it matches"
    assert "not being enforced" in " ".join(r.getMessage()
                                            for r in caplog.records)


def test_an_unlisted_band_falls_back_to_the_instrument_default():
    assert required_flat_type_for("NB2.108", flat_types=FLAT_TYPES,
                                  default=DEFAULT_FLAT_TYPE) == "LAMP"


# --- the audit trail reaches the product -------------------------------

def _reduce(science, flat, **kwargs):
    """Reduce one frame against one flat, with no dark.

    ITIME and COADDS are both 1 in the helper frames, so the normal
    divisions run and change nothing -- the frames go through the same path
    as real data rather than a special-cased one.
    """
    return reduce_frame(science, [flat], [], **kwargs)


def test_a_checked_reduction_records_that_it_was_checked():
    """"Was this rule enforced?" has to be answerable from the file, not
    from whoever happened to run it."""
    reduced = _reduce(_science(), _flat(),
                      required_flat_types=FLAT_TYPES,
                      default_required_flat_type=DEFAULT_FLAT_TYPE)
    assert reduced["FLATCHK"] is True
    assert reduced["FLATMISM"] is False


def test_a_skipped_check_is_recorded_on_the_reduced_frame():
    """Without the instrument table the requirement cannot be evaluated. The
    frame has to say so: a log line has scrolled away by the time anyone
    asks."""
    reduced = _reduce(_science(), _flat())
    assert reduced["FLATCHK"] is False
    assert "flat_checked=F" in describe(reduced)


def test_an_overridden_mismatch_is_recorded_on_the_reduced_frame():
    """Previously FLATMISM was stamped on a copy of the flat that
    reduce_frame threw away, so the product carried no trace of it."""
    reduced = _reduce(_science("Lp + Wollaston", "Lp"),
                      _flat("Lp + Wollaston", "Lp", "LAMP"),
                      required_flat_types=FLAT_TYPES,
                      default_required_flat_type=DEFAULT_FLAT_TYPE,
                      allow_flat_type_mismatch=True)
    assert reduced["FLATMISM"] is True
    assert reduced["FLATCHK"] is True


# --- instrument-supplied defaults ---------------------------------------

def _raw_flat(hwp, band="Kp", n=8):
    """A raw dome flat at one HWP angle, for the polarimetric split."""
    return Frame(np.ones((n, n)),
                 {"FILTER": f"{band} + Wollaston", "FWINAME": band,
                  "NAXIS1": n, "NAXIS2": n, "ITIME": 1.0, "COADDS": 1,
                  "SAMPMODE": 3, "READS": 1, "PCUPR": hwp,
                  "FILENAME": f"f{hwp}.fits"})


def _critical_angle_flats():
    """Three flats at each critical angle -- a polarimetric set."""
    return [_raw_flat(a) for a in NIRC2PolarimetryData.critical_angles
            for _ in range(3)]


def test_instrument_supplies_the_polarimetric_split():
    """Passing instrument= must do what naming the two keywords did, or the
    convenience is a trap: the flats silently stop being ranked."""
    flats, _ = make_master_flats(_critical_angle_flats(), [], [], [],
                                 instrument=NIRC2PolarimetryData())
    assert any(f.get("POLFLAT") for f in flats), \
        "critical-angle flats were not recognised as polarimetric"


def test_an_explicit_argument_still_beats_the_instrument():
    """The instrument is a default, not an override."""
    flats, _ = make_master_flats(_critical_angle_flats(), [], [], [],
                                 instrument=NIRC2PolarimetryData(),
                                 modulator_keyword="NOSUCHKEY")
    assert not any(f.get("POLFLAT") for f in flats), \
        "the explicit keyword was ignored in favour of the instrument"


def test_instrument_supplies_the_flat_type_table():
    """The dangerous one to forget: without the table the band requirement
    is not enforced at all."""
    science = _science("Lp + Wollaston", "Lp")
    lamp = _flat("Lp + Wollaston", "Lp", "LAMP")

    # instrument= alone must still enforce SKY for L-prime
    with pytest.raises(ValueError, match="requires a SKY flat"):
        find_closest_flat(science, [lamp],
                          required_flat_types=NIRC2PolarimetryData.required_flat_types,
                          default_required_flat_type=(
                              NIRC2PolarimetryData.default_required_flat_type))


def test_instrument_supplies_the_bad_pixel_mask(monkeypatch):
    """It is a method that reads a FITS file, so it must be called only when
    it is actually needed."""
    calls = []

    class Counting(NIRC2PolarimetryData):
        def bad_pixel_mask(self):
            calls.append(1)
            return np.zeros((8, 8), dtype=bool)

    inst = Counting()
    darks = [_raw_flat(0.0) for _ in range(3)]
    make_master_darks(darks, instrument=inst)
    assert calls, "the instrument's mask was never consulted"

    calls.clear()
    make_master_darks(darks, instrument=inst,
                      bad_pixel_mask=np.zeros((8, 8), dtype=bool))
    assert not calls, "the mask was read despite one being supplied"


# --- a missing flat refuses, and the two deliberate ways past it --------

def test_a_missing_flat_refuses_by_default():
    """Dividing by ones leaves the detector response in the data and is easy
    to miss later, so it must not be the quiet path."""
    with pytest.raises(ValueError) as excinfo:
        _find_flat(_science("Lp + Wollaston", "Lp"), [_flat()])
    message = str(excinfo.value)
    assert "flat_override" in message and "allow_no_flat" in message, \
        "the refusal has to say how to proceed deliberately"


def test_allow_no_flat_reduces_unflattened():
    """The old behaviour, now opt-in."""
    _, got = _find_flat(_science("Lp + Wollaston", "Lp"), [_flat()],
                        allow_no_flat=True)
    assert got is None


def test_flat_override_uses_the_named_flat_whatever_its_filter():
    """The per-reduction replacement for a flat-exceptions entry: the user
    names the flat, the pipeline never guesses one."""
    kp_flat = _flat("Kp + Wollaston", "Kp", "LAMP")
    _, got = _find_flat(_science("H2O_ice + Wollaston", "H2O_ice"),
                        [kp_flat], flat_override=kp_flat)
    assert got is not None and "Kp" in got["FILTER"]
    assert got["FLATSUB"] is True, "the substitution must be recorded"


def test_an_overridden_flat_still_faces_the_type_check():
    """Choosing the filter by hand must not quietly waive the SKY/LAMP rule
    as well -- that is a separate decision."""
    lamp = _flat("Kp + Wollaston", "Kp", "LAMP")
    with pytest.raises(ValueError, match="requires a SKY flat"):
        _find_flat(_science("Lp + Wollaston", "Lp"), [lamp],
                   flat_override=lamp)


def test_substitution_reaches_the_reduced_frame():
    kp_flat = _flat("Kp + Wollaston", "Kp", "LAMP")
    reduced = _reduce(_science("H2O_ice + Wollaston", "H2O_ice"), kp_flat,
                      flat_override=kp_flat)
    assert reduced["FLATSUB"] is True
    assert reduced["FLATDIV"] is True
