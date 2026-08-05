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

# Which kind of flat is best depends on the band. In the thermal infrared
# the dome lamp is swamped by thermal background, so sky flats are used;
# in the near infrared the lamp flats are cleaner. Users can override this
# per reduction (some prefer skies for JHK too).
PREFERRED_FLAT_TYPE_BY_BAND = {
    "Lp": "SKY", "L": "SKY", "Ms": "SKY", "M": "SKY",
    "J": "LAMP", "H": "LAMP", "K": "LAMP", "Kp": "LAMP", "Ks": "LAMP",
}
DEFAULT_PREFERRED_FLAT_TYPE = "LAMP"

RANKED_SKIES_KEYLISTS = [
    ["FILTER", "ITIME", "COADDS"],
    ["FILTER", "ITIME"],
    ["FILTER"],
]
