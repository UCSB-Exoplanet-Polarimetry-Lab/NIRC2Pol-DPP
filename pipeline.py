"""High-level pipeline interface: pulls individual steps into a "recipe".

A ``Pipeline`` is a list of named steps sharing a context dict. Each step
is a plain function taking the context and returning a result, which is
stored back into the context under the step's name — so later steps can use
earlier results, and after ``run()`` the user can inspect any intermediate
product::

    pipe = Pipeline({"paths": paths, "instrument": nirc2.NIRC2PolarimetryData()})
    pipe.add_step("sort", lambda ctx: ctx["instrument"].sort_frames(raw_files))
    pipe.add_step("masters", make_masters_step)
    ...
    pipe.run()
    reduced = pipe.context["reduce"]

Every step is optional and replaceable: for a custom reduction, skip the
orchestrator entirely and call the module functions directly (see
examples/process_polmode.py).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, context=None):
        self.context = dict(context or {})
        self.steps = []

    def add_step(self, name, func, **kwargs):
        """Append a step. ``func(context, **kwargs)`` runs when the pipeline
        reaches it; its return value lands in ``context[name]``."""
        self.steps.append((name, func, kwargs))
        return self

    def run(self, from_step=None):
        """Run all steps in order (optionally starting at ``from_step``,
        e.g. to resume after tweaking a parameter)."""
        started = from_step is None
        for name, func, kwargs in self.steps:
            if not started:
                if name == from_step:
                    started = True
                else:
                    log.info("Skipping step %s", name)
                    continue
            log.info("Running step %s...", name)
            self.context[name] = func(self.context, **kwargs)
        return self.context
