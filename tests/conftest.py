"""Synthetic dual-beam polarimetry, for tests that need no real data.

Everything here is built from a forward model with *known* answers — a
disk with a known azimuthal polarization, a known I -> Q/U leakage and a
known fast axis offset — so the reduction can be checked against truth
rather than against its own previous output.

The forward model runs the reduction backwards:

1. start from radial Stokes: ``Q_phi = amplitude(r)``, ``U_phi = 0``
   (tangential polarization, what single scattering off a disk produces);
2. convert to sky Q/U by inverting SPIE Eq. 6;
3. rotate into the instrument frame by ``-theta_rot``, the inverse of what
   ``rotate_qu`` will later undo;
4. add the leakage, ``Q += ipq*I`` and ``U += ipu*I``;
5. turn Q/U into per-HWP-angle single differences and split those into two
   beams.

``double_difference`` then has to give step 4 back, and ``build_stokes_cube``
step 2, if and only if it is handed the injected ``theta_off``.
"""

import numpy as np
import pytest

from instruments.base import PolarimetryData
from utils.frame import Frame

NY = NX = 96
CRITICAL = (0.0, 45.0, 22.5, 67.5)


class SyntheticPolarimetryData(PolarimetryData):
    """Minimal instrument: two beams stacked vertically in one array."""

    name = "synthetic"
    plate_scale = 0.01
    modulator_keyword = "HWP"
    critical_angles = CRITICAL
    beam_height = NY
    background_method = None      # synthetic frames carry no pedestal

    def bad_pixel_mask(self):
        return np.zeros((2 * NY, NX), dtype=bool)

    def gain(self, header):
        return 1.0

    def saturation_limit(self, header):
        return 1e12

    def sort_frames(self, filenames, **kwargs):
        return {"sci": list(filenames), "darks": [], "flats": [],
                "flats_sky": [], "flats_lampon": [], "flats_lampoff": []}

    def north_angle(self, header):
        return float(header.get("NORTH", 0.0))

    def split_beams(self, frame):
        data = frame.data if hasattr(frame, "data") else np.asarray(frame)
        return np.stack([data[:NY], data[NY:]])

    def qu_rotation_angle(self, header, fast_axis_offset=0.0):
        """Same shape as the NIRC2 model, so the factor of 4 is exercised."""
        if fast_axis_offset is None:
            fast_axis_offset = self.fast_axis_offset
        return (-2.0 * header["PARANG"] + 2.0 * header["EL"]
                + 2.0 * header["ROT"] + 4.0 * fast_axis_offset)

    def occulting_radius(self, header):
        return header.get("OCCRAD")


def disk_radial_stokes(shape=(NY, NX), r_inner=12.0, r_outer=34.0,
                       amplitude=1.0):
    """``(Q_phi, U_phi, I)`` for a tangentially polarized annular disk."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - cy, xx - cx)
    phi = np.arctan2(yy - cy, xx - cx)

    ring = ((r >= r_inner) & (r <= r_outer)).astype(float)
    q_phi = amplitude * ring
    u_phi = np.zeros_like(q_phi)
    # a bright, smooth stellar halo so I is positive everywhere and the
    # mask-edge annulus has something unpolarized to measure
    I = 1000.0 * np.exp(-r ** 2 / (2 * 22.0 ** 2)) + 10.0
    return q_phi, u_phi, I, phi


def synth_cycle(theta_off, ipq=0.0, ipu=0.0, parang=0.0, el=45.0, rot=0.0,
                north=0.0, amplitude=1.0, seed=None, noise=0.0):
    """One HWP cycle of four synthetic frames with known truth."""
    q_phi, u_phi, I, phi = disk_radial_stokes(amplitude=amplitude)

    # radial -> sky Q/U (inverse of SPIE Eq. 6)
    sky_q = q_phi * np.cos(2 * phi) - u_phi * np.sin(2 * phi)
    sky_u = q_phi * np.sin(2 * phi) + u_phi * np.cos(2 * phi)

    header0 = {"PARANG": parang, "EL": el, "ROT": rot, "NORTH": north}
    inst = SyntheticPolarimetryData()
    theta = np.radians(inst.qu_rotation_angle(header0, theta_off)
                       + 2.0 * north)

    # sky -> instrument frame (inverse of rotate_qu)
    q_i = sky_q * np.cos(theta) - sky_u * np.sin(theta)
    u_i = sky_q * np.sin(theta) + sky_u * np.cos(theta)

    # the leakage the reduction is supposed to find
    q_i = q_i + ipq * I
    u_i = u_i + ipu * I

    rng = np.random.default_rng(seed)
    frames = []
    for angle, d in zip(CRITICAL, (q_i, -q_i, u_i, -u_i)):
        s = I.copy()
        dd = d.copy()
        if noise:
            dd = dd + rng.normal(0, noise, size=d.shape)
        bottom, top = 0.5 * (s - dd), 0.5 * (s + dd)
        data = np.concatenate([bottom, top], axis=0)
        header = dict(header0, HWP=angle, OCCRAD=12.0)
        frames.append(Frame(data, header))
    return frames


@pytest.fixture
def instrument():
    return SyntheticPolarimetryData()


@pytest.fixture
def truth():
    """The injected values every test checks against."""
    return {"theta_off": 7.5, "ipq": 0.021, "ipu": -0.013}


@pytest.fixture
def cycles(truth):
    """Four cycles at different parallactic angles, as on sky."""
    return [synth_cycle(truth["theta_off"], truth["ipq"], truth["ipu"],
                        parang=pa, el=45.0 + 0.5 * i, rot=10.0 * i)
            for i, pa in enumerate((-20.0, -5.0, 10.0, 25.0))]


@pytest.fixture
def clean_cycles(truth):
    """Same, with no instrumental leakage injected."""
    return [synth_cycle(truth["theta_off"], 0.0, 0.0,
                        parang=pa, el=45.0 + 0.5 * i, rot=10.0 * i)
            for i, pa in enumerate((-20.0, -5.0, 10.0, 25.0))]
