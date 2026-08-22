"""Write pipeline data products to disk, with provenance headers.

Covers the outputs of the block diagram — intermediate products (reduced
frames, per-HWP-cycle Stokes cubes) and final products (median Stokes cube,
PI / AoLP / DoLP, radial Stokes) — through one object whose output location
the user sets in a single place::

    writer = ProductWriter("/path/to/my/output", target="HD_377")
    writer.save_reduced(reduced_frames)
    writer.save_stokes_cycles(cubes, cycles)
    writer.save_median_stokes(median_cube)
    writer.save_derived_products(median_cube)

Layout under ``output_dir``::

    <target>_reduced/            one FITS per reduced science frame
    <target>_stokes_cycles/      one [I,Q,U] cube per HWP cycle, each with
                                 its own cycle's header
    <target>_median_stokes.fits  median-combined [I, Q, U]
    <target>_PI.fits, _AoLP.fits, _DoLP.fits
    <target>_Qphi.fits, _Uphi.fits
"""

from __future__ import annotations

import logging
import os

import numpy as np

from nirc2pol.utils import Frame
from nirc2pol.utils.provenance import record_step

log = logging.getLogger(__name__)


class ProductWriter:
    """Writes pipeline products under ``output_dir``.

    Parameters
    ----------
    output_dir : str
        Where everything goes; created if needed. This is the single place
        a user changes to redirect all output.
    target : str
        Prefix for product filenames.
    overwrite : bool
        Overwrite existing files (default True).
    """

    def __init__(self, output_dir, target="target", overwrite=True):
        """Create a writer rooted at one output directory.

        Parameters
        ----------
        output_dir : str
            Directory for the products; created if absent.
        target : str, optional
            Prefix for every filename, so several targets can share a directory.
        overwrite : bool, optional
            Replace existing files.
        """
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.target = target
        self.overwrite = overwrite
        os.makedirs(self.output_dir, exist_ok=True)
        log.info("Products will be written to %s", self.output_dir)

    # -- helpers ----------------------------------------------------------

    def path(self, name):
        """Absolute path of a product file inside the output directory."""
        return os.path.join(self.output_dir, f"{self.target}_{name}")

    def _save(self, data, header, name, step=None, bunit=None, **params):
        """Write one product, stamping its provenance.

        Parameters
        ----------
        data : ndarray
            Array to write.
        header : Header or None
            Header to carry across.
        name : str
            Product name; the filename becomes ``<target>_<name>``.
        step : str, optional
            Provenance step recorded via ``utils.provenance.record_step``.
        bunit : str, optional
            Override the units inherited from ``header``. Pass ``""`` for a
            dimensionless product; None leaves whatever the header carried.
        **params
            Extra provenance parameters.

        Returns
        -------
        str
            The path written.
        """
        frame = Frame(data, header.copy() if header is not None else None)
        if step:
            record_step(frame, step, **params)
        frame["PRODUCT"] = (name, "NIRC2Pol-DPP product type")
        if bunit is not None:
            # "" is meaningful here: a dimensionless ratio, as opposed to
            # None which means "whatever the input header already said".
            frame["BUNIT"] = (bunit, "physical units of the array values")
        out = self.path(name if name.endswith(".fits") else f"{name}.fits")
        frame.save(out, overwrite=self.overwrite)
        log.info("wrote %s", out)
        return out

    # -- intermediate products --------------------------------------------

    def save_reduced(self, frames, subdir=None):
        """Write dark-subtracted / flat-fielded science frames.

        Parameters
        ----------
        frames : iterable of Frame
            Reduced science frames.
        subdir : str, optional
            Subdirectory to write into.

        Returns
        -------
        list of str
            Paths written.
        """
        folder = os.path.join(self.output_dir,
                              subdir or f"{self.target}_reduced")
        os.makedirs(folder, exist_ok=True)
        paths = []
        for i, f in enumerate(frames):
            name = f.get("RED-FN") or f"reduced_{i:04d}.fits"
            out = os.path.join(folder, name)
            f.save(out, overwrite=self.overwrite)
            paths.append(out)
        log.info("wrote %d reduced frames to %s", len(paths), folder)
        return paths

    def save_stokes_cycles(self, cubes, cycles=None, header=None, **params):
        """Write one Stokes cube file per HWP cycle.

        ``cubes`` is ``(ncycles, 3, ny, nx)``, written as ``ncycles`` files
        in ``<target>_stokes_cycles/``. Separate files rather than one
        stacked array because each keeps its **own** cycle's header -- that
        cycle's PARANG, EL and ROTPDEST. A stacked cube can carry only one
        header, which would describe the first cycle and misdescribe every
        other, and each cycle sits at a different point on the sky.

        Parameters
        ----------
        cubes : ndarray
            ``(ncycles, 3, ny, nx)`` per-cycle cubes.
        cycles : list of list of Frame, optional
            The cycles they came from; each file takes the header of its own
            cycle's first frame. Without them every file falls back to
            ``header``, which is the same header on all of them.
        header : Header, optional
            Reduction-level header. Keywords it carries that the cycle's own
            header does not -- ``THETAOFF``, say -- are copied in; where both
            carry one, the cycle's own value wins, since that is the value
            describing this cube. Commentary cards (HISTORY, COMMENT) are not
            copied: ``record_step`` writes this product's own.
        **params
            Extra provenance parameters.

        Returns
        -------
        list of str
            The paths written, in cycle order.
        """
        cubes = np.asarray(cubes)
        folder = os.path.join(self.output_dir, f"{self.target}_stokes_cycles")
        os.makedirs(folder, exist_ok=True)

        cycle_paths = []
        for i, cube in enumerate(cubes):
            if cycles:
                hdr = cycles[i][0].header.copy()
                for card in (header.cards if header is not None else ()):
                    key = card.keyword
                    if key and key not in ("HISTORY", "COMMENT") \
                            and key not in hdr:
                        hdr[key] = (card.value, card.comment)
            else:
                hdr = header.copy() if header is not None else None

            frame = Frame(cube, hdr)
            record_step(frame, "stokes cube for one HWP cycle",
                        cycle=i, ncycles=len(cubes),
                        axes="([I,Q,U], y, x)", **params)
            frame["POLCYCLE"] = (i, "HWP cycle index")
            frame["POLNCYC"] = (len(cubes), "HWP cycles in this reduction")
            frame["PRODUCT"] = ("stokes_cycle", "NIRC2Pol-DPP product type")
            out = os.path.join(folder,
                               f"{self.target}_stokes_cycle_{i:03d}.fits")
            frame.save(out, overwrite=self.overwrite)
            cycle_paths.append(out)

        log.info("wrote %d per-cycle Stokes cubes to %s",
                 len(cycle_paths), folder)
        return cycle_paths

    # -- final products ----------------------------------------------------

    def save_median_stokes(self, cube, header=None, **params):
        """Write the median-combined [I, Q, U] Stokes cube.

        Parameters
        ----------
        cube : ndarray
            ``(3, ny, nx)`` median-combined cube.
        header : Header, optional
            Header carried onto the product.
        **params
            Extra provenance parameters.

        Returns
        -------
        str
            Path written.
        """
        return self._save(np.asarray(cube), header, "median_stokes",
                          step="median-combined Stokes cube", **params)

    def save_derived_products(self, cube, header=None, center=None, **params):
        """Write PI, AoLP, DoLP and the radial Stokes images derived from a
                ``(3, ny, nx)`` Stokes cube.

        Parameters
        ----------
        cube : ndarray
            ``(3, ny, nx)`` Stokes cube.
        header : Header, optional
            Header carried onto the products.
        center : tuple of float, optional
            Centre for the radial Stokes.
        **params
            Extra provenance parameters.

        Returns
        -------
        list of str
            Paths written.
        """
        from .stokes import polarization_products, radial_stokes

        cube = np.asarray(cube)
        pi, aolp, dolp = polarization_products(cube)
        q_phi, u_phi = radial_stokes(cube[1], cube[2], center=center)

        out = {}
        # BUNIT arrives on the header copied from a reduced frame, where it
        # describes the Stokes values. It is right for PI and the radial
        # Stokes, which share those units, and wrong for the two derived
        # quantities that do not: DoLP is a ratio and AoLP is an angle. Each
        # product therefore states its own.
        for name, data, step, extra in [
                ("PI", pi, "polarized intensity sqrt(Q^2+U^2)", {}),
                ("AoLP", aolp, "angle of linear polarization 0.5*atan2(U,Q)",
                 {"units": "deg"}),
                ("DoLP", dolp, "degree of linear polarization PI/I", {}),
                ("Qphi", q_phi, "radial Stokes Q_phi", {"center": center}),
                ("Uphi", u_phi, "radial Stokes U_phi", {"center": center})]:
            unit = {"AoLP": "deg", "DoLP": ""}.get(name)
            out[name] = self._save(data, header, name, step=step, bunit=unit,
                                   **{**extra, **params})
        return out
