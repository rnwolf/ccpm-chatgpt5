# CCPM Tool ala ChatGPT-5

## Key features

compute_asap(tasks, calendar) — forward earliest (ASAP) start/finish using working-day arithmetic.

compute_alap(tasks, calendar, delivery_day) — backward latest (ALAP) start/finish anchored to a delivery day, respecting working days.

resource_constrained_alap(...) — schedule within ALAP windows while enforcing per-resource daily availability; fails with clear error if impossible.

schedule_tasks(...) — wrapper that runs ASAP → ALAP → resource-constrained scheduling iteratively until stable (or max iterations reached). Protects against infinite loops.

Utility functions for working-day arithmetic (add_working_days_forward, subtract_working_days_backward, next_working_day, prev_working_day).

## Usage notes


The ALAP/ASAP passes use project working days — the module builds a working-day set from the calendar. If your project horizon is much larger than 365 days, increase max_day where appropriate.

resource_constrained_alap assumes single-capacity resources (one unit per day). If you need multi-capacity resources, extend Resource to include units_per_day and change allocation checks accordingly.

The scheduler currently places tasks as late as possible inside their ALAP windows (this is CCPM-friendly). You can alter priority ordering if you want different heuristics (e.g., shortest-duration-first, or including feed priority).

Defensive checks so that scheduling passes fail with clear errors instead of hanging in an infinite loop.

## Next Steps

implement compute_critical_chain() (if you want chain detection used to create project and feeding buffers),

add buffer insertion routines (we sketched those earlier),

add multi-unit resource capacity,

add visualization / Gantt export.

add multi-capacity resource support, or

wire in the feeding/project buffer insertion and show a complete example with buffers inserted and schedule re-run.
