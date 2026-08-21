"""Run the standard reduction from a script instead of the command line.

``nirc2pol-reduce my_night.toml`` does exactly what this file does. The
script form earns its place in what comes *after* the call: ``run`` hands
back everything it built, still in memory, so you can look at any of it or
carry on with something the recipe does not do.

    nirc2pol-reduce --template > my_night.toml   # every option and default
    python process_polmode.py my_night.toml

With no argument it falls back to ``examples/reduction_config.toml``, the
template checked in beside this file.

There is one copy of the reduction itself, in :func:`nirc2pol.polmode.run`,
which the command line and this script both call. For a reduction that
departs from that recipe, call the module functions yourself --
``examples/tutorial.ipynb`` walks through them one at a time and explains
what each is for.

No ``sys.path`` juggling: the pipeline is installed (``pip install -e .``),
so this runs from any directory.
"""

import logging
import os
import sys

from nirc2pol.polmode import run
from nirc2pol.reduction.config import ReductionConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CONFIG_PATH = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "reduction_config.toml"))

cfg = ReductionConfig.from_toml(CONFIG_PATH)
products = run(cfg, config_path=CONFIG_PATH)

# Everything the run built. products["cycles"] holds the matched HWP cycles,
# products["stokes_cubes"] one cube each, and products["median_cube"] the
# combination -- inspect them, plot them, or feed them onward from here.
writer, run_log = products["writer"], products["run_log"]

print(f"\nDone: {len(products['stokes_cubes'])} HWP cycles -> "
      f"{products['median_cube'].shape} median Stokes cube in "
      f"{writer.output_dir}")
print(f"Fast axis offset: {products['theta_off']} deg")
print(f"Leakage:          {products['ip']}")
print(f"Log:              {run_log.path} ({run_log.warnings} warnings)")
