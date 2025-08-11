# CCPM Tool

Experiment to see if we can get AI coders to build a python CCPM schedeuling tool.


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

## Development Next Steps


### Phase 1: Fix Critical Bugs and Stabilize Core Logic

1. **Fix Resource Modeling**:

The `Resource` class is missing a mechanism to track its allocations over time. The `resource_constrained_alap` function tries to use an `allocations` attribute on the `Resource` object, which will cause a crash. I will add an `allocations` dictionary to the `Resource` class to fix this fundamental bug.

2. **Consolidate Calendar Logic**:

There's a mix of standalone functions and class methods for calendar calculations. This can lead to confusion and inconsistent behavior. I will consolidate the calendar logic into the `ProjectCalendar` class and update the rest of the code to use the methods from the class. This will provide a single, reliable source for all date-related calculations.

3. **Introduce Unit Testing**:

To ensure the core components are working correctly and to prevent future bugs, I will set up a basic testing file (`test_ccpm.py`). I'll start by adding tests for the calendar and resource logic. Do you have a preferred testing framework like `pytest`, or should I use Python's standard `unittest` library?

### Phase 2: Verify Scheduling and Implement CCPM

1. **Verify Scheduling Algorithms**:

Once the core components are stable and tested, I will add specific tests for the scheduling algorithms (`compute_asap`, `compute_alap`, and `resource_constrained_alap`). Using simple, predictable project networks, we can verify that they produce correct schedules.

2. **Implement Critical Chain Identification**:

With a reliable resource-constrained schedule, I will implement the logic to identify the critical chain. This involves finding the longest path of resource-dependent tasks and setting the `on_critical_chain` flag on those tasks.

3. **Implement Buffering**:

Finally, I will add the core CCPM feature of buffer insertion. This includes writing functions to calculate buffer sizes (as a percentage of the chain's duration) and to insert `ProjectBuffer` and `FeedingBuffer` tasks into the project network with the correct dependencies.

This phased approach will allow us to systematically debug the code, build a solid and testable foundation, and then implement the full set of CCPM features.

### Phase 3 Enhancements

add multi-unit resource capacity,

add visualization / Gantt export.

add multi-capacity resource support, or
