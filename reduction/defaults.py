"""Default header keylists used for matching calibration frames to science
frames.

These defaults are the standard NIRC2/Keck keywords (they happen to be common
FITS keywords, so they work for many instruments), but every function that
uses them accepts an explicit keylist so other instruments can override.
"""

# keywords that must match for a dark to pair with a frame, from most to
# least strict — see reduction.calibrate.find_closest_dark
DARKS_KEYLIST = ["NAXIS1", "NAXIS2", "ITIME", "COADDS", "SAMPMODE", "READS"]

RANKED_DARKS_KEYLISTS = [
    ["NAXIS1", "NAXIS2", "ITIME", "COADDS", "SAMPMODE", "READS"],
    ["NAXIS1", "NAXIS2", "ITIME", "COADDS", "SAMPMODE"],
    ["NAXIS1", "NAXIS2", "ITIME", "COADDS"],
    ["NAXIS1", "NAXIS2", "ITIME"],
    ["ITIME"],
]

FLATS_KEYLIST = ["NAXIS1", "NAXIS2", "ITIME", "COADDS", "FILTER", "SAMPMODE", "READS"]

FLATS_MATCH_KEYLIST = ["FILTER"]

# Flat-to-frame matching. The filter MUST match: a flat in the wrong filter
# describes the wrong throughput pattern. Detector size ideally matches, but
# a larger flat can simply be trimmed to the frame, so it is only a
# preference. Exposure settings are irrelevant -- the flat is normalized.
RANKED_FLATS_KEYLISTS = [
    ["FILTER", "NAXIS1", "NAXIS2"],   # exact size: no trimming needed
    ["FILTER"],                        # any size: larger flats get cropped
]

# NB: which flat type a band requires is NOT here. It differs per
# instrument, so it lives on the instrument (see
# PolarimetryData.required_flat_types) and is passed in by the caller.

RANKED_SKIES_KEYLISTS = [
    ["FILTER", "ITIME", "COADDS"],
    ["FILTER", "ITIME"],
    ["FILTER"],
]
