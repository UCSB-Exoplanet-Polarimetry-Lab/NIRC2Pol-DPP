"""Default header keylists used for matching calibration frames to science
frames.

These defaults are the standard NIRC2/Keck keywords (they happen to be common
FITS keywords, so they work for many instruments), but every function that
uses them accepts an explicit keylist so other instruments can override.
"""

# Keywords that must match for a dark to pair with a frame. All of them:
# dark current scales with ITIME and COADDS, and the bias structure left
# behind depends on the sampling mode and how many reads went into it, so a
# dark taken any other way is the wrong dark. Size is included because a
# subarray reads out differently from the full frame.
#
# This list used to be the first of five, each dropping another keyword,
# ending at ["ITIME"] alone -- which accepted a 512x512 CDS dark with one
# coadd in place of a 1024x1024 MCDS dark with 45. Substituting the wrong
# dark does not fail loudly, it just subtracts the wrong pedestal, so there
# is no relaxation now.
# Dark matching relaxes down this ladder until something matches, following
# AIR.jl. The first entry is the only one that is physically right: dark
# current scales with ITIME and COADDS, and the bias structure depends on the
# sampling mode and the number of reads. Everything below it trades a known
# error for an answer, so find_closest_dark warns on every rung but the first,
# and warns again if even ITIME cannot be matched.
RANKED_DARKS_KEYLISTS = [
    ["NAXIS1", "NAXIS2", "ITIME", "COADDS", "SAMPMODE", "READS"],
    ["NAXIS1", "NAXIS2", "ITIME", "COADDS", "SAMPMODE"],
    ["NAXIS1", "NAXIS2", "ITIME", "COADDS"],
    ["NAXIS1", "NAXIS2", "ITIME"],
    ["ITIME"],
]

DARKS_KEYLIST = RANKED_DARKS_KEYLISTS[0]

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
