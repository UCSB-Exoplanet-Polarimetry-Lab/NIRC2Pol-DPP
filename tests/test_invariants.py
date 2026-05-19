"""Architectural invariant: science layer must not import from instruments/nirc2/.

Imports every module under reduction/, polarimetry/, calibration/, and
pipeline.py and asserts that none of them transitively import from
nirc2pol.instruments.nirc2.
"""
