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

The bare instrument above is enough to sort frames. Any step that splits the
Wollaston beams also needs the beam geometry, which is read per epoch from
instruments/nirc2.ini and will refuse rather than guess if that epoch is not
recorded.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Pipeline:
    """A reduction assembled from named steps sharing a context.

    Each step is a plain function of the context dict; its return value is
    stored back under the step's name, so later steps can use earlier results
    and every intermediate stays inspectable after the run.

    The orchestrator is entirely optional -- see ``examples/process_polmode.py``
    for the same reduction written as direct calls. It earns its place when you
    want to re-run part of a reduction after changing a parameter, via
    ``run(from_step=...)``.

    Attributes
    ----------
    context : dict
        Shared state. Seeded by the caller and extended with each step's result.
    steps : list of tuple
        ``(name, func, kwargs)`` in execution order.
    """
    def __init__(self, context=None):
        """Create a pipeline.

        Parameters
        ----------
        context : dict, optional
            Initial shared state, e.g. the instrument and paths. Copied, so the
            caller's dict is not mutated.
        """
        self.context = dict(context or {})
        self.steps = []

    def add_step(self, name, func, **kwargs):
        """Append a step. ``func(context, **kwargs)`` runs when the pipeline
                reaches it; its return value lands in ``context[name]``.

        Append a step.

        Parameters
        ----------
        name : str
            Key under which the result is stored, and the handle for
            ``run(from_step=...)``.
        func : callable
            Called as ``func(context, **kwargs)`` when the pipeline reaches it.
        **kwargs
            Extra arguments bound to this step.

        Returns
        -------
        Pipeline
            ``self``, so steps can be chained.
        """
        self.steps.append((name, func, kwargs))
        return self

    def run(self, from_step=None):
        """Run all steps in order (optionally starting at ``from_step``,
                e.g. to resume after tweaking a parameter).

        Run the steps in order.

        Parameters
        ----------
        from_step : str, optional
            Skip everything before this step, reusing whatever the context already
            holds. Useful for re-running the tail of a reduction after changing a
            parameter, without repeating the expensive front.

        Returns
        -------
        dict
            The context, with each step's result under its name.
        """
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
