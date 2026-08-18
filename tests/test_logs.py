"""The per-reduction log file."""

import datetime
import logging
import os

import pytest

from utils import ObslogPaths, start_reduction_log


def _lines(path):
    with open(path) as handle:
        return [line.rstrip("\n") for line in handle]


def test_timestamps_are_utc_not_local_wearing_a_z(tmp_path):
    """logging.Formatter renders asctime with time.localtime unless told
    otherwise, so the Z in the format string was a lie on any machine not set
    to UTC -- which is how the header said 22:40 and the lines said 15:40 in
    the same file."""
    path = str(tmp_path / "r.log")
    with start_reduction_log(path):
        logging.getLogger("probe").warning("a line to timestamp")

    line = next(l for l in _lines(path) if "probe" in l)
    stamp = datetime.datetime.strptime(
        line.split()[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    drift = abs((datetime.datetime.now(datetime.timezone.utc)
                 - stamp).total_seconds())
    assert drift < 30, (
        "the timestamp is offset from UTC, so the Z is wrong -- on a machine "
        "in UTC this test passes either way, which is why it compares "
        "against an aware now() rather than trusting the suffix")


def test_the_footer_counts_the_warnings_that_went_past(tmp_path):
    """The count answers 'is there anything in here I need to read?', so it
    has to match what is actually in the file."""
    path = str(tmp_path / "r.log")
    with start_reduction_log(path) as run_log:
        logging.getLogger("probe").info("not a warning")
        logging.getLogger("probe").warning("one")
        logging.getLogger("probe").warning("two")
        assert run_log.warnings == 2

    lines = _lines(path)
    assert sum(1 for l in lines if " WARNING " in l) == 2
    assert "2 warning(s)" in lines[-1]


def test_a_clean_run_says_so(tmp_path):
    path = str(tmp_path / "r.log")
    with start_reduction_log(path):
        logging.getLogger("probe").info("all fine")
    assert "no warnings" in _lines(path)[-1]


def test_settings_records_the_configuration(tmp_path):
    """The body records what happened; this records what was asked for."""
    path = str(tmp_path / "r.log")
    with start_reduction_log(path) as run_log:
        run_log.settings(theta_off=-13.0, register_method="min")

    text = "\n".join(_lines(path))
    assert "setting theta_off = -13.0" in text
    assert "setting register_method = min" in text


def test_finish_detaches_and_restores_the_root_logger(tmp_path):
    """It raises the root level to capture INFO, so failing to put it back
    would silently change what every later run prints."""
    root = logging.getLogger()
    before_level, before_handlers = root.level, len(root.handlers)

    run_log = start_reduction_log(str(tmp_path / "r.log"))
    assert len(root.handlers) == before_handlers + 1
    run_log.finish()

    assert root.level == before_level
    assert len(root.handlers) == before_handlers


def test_finish_is_idempotent(tmp_path):
    path = str(tmp_path / "r.log")
    run_log = start_reduction_log(path)
    run_log.finish()
    run_log.finish()
    assert sum(1 for l in _lines(path) if l.startswith("# finished")) == 1


def test_nothing_written_after_finish_lands_in_the_file(tmp_path):
    path = str(tmp_path / "r.log")
    run_log = start_reduction_log(path)
    run_log.finish()
    logging.getLogger("probe").warning("after the end")
    assert not any("after the end" in l for l in _lines(path))


def test_the_log_sits_beside_the_products_and_carries_the_date():
    paths = ObslogPaths("/data", "2025-12-08")
    name = os.path.basename(paths.log_file)
    assert name == "reduction_2025-12-08.log"
    assert os.path.dirname(paths.log_file) == paths.data_folder
