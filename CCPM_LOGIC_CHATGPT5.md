# logic for creating a Critical Chain Project Schedule (CCPM)

In plain English, starting from the last task in the project and working backwards.
I’ll describe it in a way that can later be translated into a Python algorithm.

## 1. Understand the Inputs

Before we start, we need:

- Task list – every activity in the project.
- Durations – preferably “aggressive but achievable” durations, not padded estimates.
- Predecessors – which tasks each task depends on.
- Resource assignments – which resource(s) each task needs.
- Resource availability – how many units of each resource are available at any time.

## 2. Start at the Final Task

We begin with the last task in the project (the one that marks completion).
We’ll work backwards through dependencies.

## 3. Build the “Latest Possible Start Times”

Working backwards:

- Set the project end date = day 0 in backwards calculation (or a given finish date).
- For the final task,
    - Latest Finish = project finish date
    - Latest Start = Latest Finish − Task Duration
- For each predecessor task,
    - Latest Finish = min(Latest Start times of all successor tasks)
    - Latest Start = Latest Finish − Task Duration
- Repeat until all tasks have latest start and latest finish times.

(At this stage we have a backward pass similar to Critical Path Method, but we have not yet handled resources.)

## 4. Identify the Critical Chain

Now we introduce CCPM’s special twist:

- Switch from dependency-driven critical path to resource-driven critical chain.
    This means:
     - If a resource is overbooked (two tasks that require it overlap), delay one so they don’t conflict.
     - Propagate delays forward to keep the sequence feasible.
- The Critical Chain = the longest chain of dependent and resource-constrained tasks from start to finish.

## 5. Remove Padding and Apply Buffers

- CCPM assumes task estimates contain safety time.
    We reduce each task’s duration to 50–70% of original estimate (aggressive but realistic).
- We then protect the schedule by inserting buffers:
    - Project Buffer – at the end of the critical chain, sized as about half the total removed safety from that chain.
    - Feeding Buffers – where a non-critical chain merges into the critical chain, sized to absorb delays before they hit the critical chain.
    - Resource Buffers – reminders to have resources ready before they’re needed.

## 6. Forward Schedule from Project Start

Once we’ve got the adjusted durations and buffers:

- Set project start date = day 0.
- Schedule tasks forward:
    - Earliest Start = max(Earliest Finish of all predecessors, availability of resources).
    - Earliest Finish = Earliest Start + Adjusted Duration.
- Place buffers in their assigned spots.
- Continue until the last task (plus project buffer) is scheduled.

## 7. Validate and Optimize

- Ensure no resource is over-allocated.
- If a delay in one task causes buffer erosion, check whether the project still completes on time.
- Adjust task overlaps or resource allocations if needed.

## Summary Flow (Backward-first approach)

- Identify last task → set Latest Finish.
- Work backward to calculate Latest Start/Finish for all tasks.
- Add resource constraints to identify the critical chain.
- Reduce task times, insert buffers.
- Schedule forward from start date with new durations and buffers.
- Validate resource availability and buffer placement.

# Aditional Notes

##  CCPM rule for handling multiple feeds

The standard CCPM method places a separate feeding buffer at the point where each feeding chain merges into the critical chain.

So in the example:

Feeding Buffer A would sit between the end of Feeding Chain A and Task X.

Feeding Buffer B would sit between the end of Feeding Chain B and Task X.

This way:

Delays in one feeding chain are absorbed in its own buffer.

They don’t immediately consume time from another chain’s buffer.

## What if the feeding chains merge before joining the critical chain?

Sometimes feeding chains join each other before hitting the critical chain. In that case:

You first place feeding buffers at the join points (if needed).

Then treat the combined merged chain as a single feeding chain into the critical chain.

Finally, place one feeding buffer at the point where that merged chain hits the critical chain.

### Automation implication

When automating:

Identify each unique path from a non-critical task to its first meeting point with the critical chain.

For each path:

Determine the total reduced (aggressive) duration.

Calculate the removed safety time.

Place a buffer sized at roughly 50% of the removed safety.

If two feeding chains merge before the critical chain, treat the merge as a new “chain start” for buffer sizing.


# Step-by-step buffer detection & placement logic

## 1. **Precompute graph helpers**

    - `topological order` of tasks (DAG assumed).
    - `reachability` map: for every node `u`, which nodes are reachable from `u` (can be computed by DFS/BFS per node or by reverse topological dynamic programming).

## 2. **Find the critical chain**

(Assume you already have this — if not: compute resource-constrained longest path or perform iterative resource-leveling after a CPM pass). Mark tasks on the critical chain (`on_critical_chain=True`).

## 3. **For each critical-chain task `C`: find feeding chain *endpoints***

    - A feeding chain endpoint is a task `t` such that:

        - `t` is **not on** the critical chain,
        - there exists a path from `t` to `C`,
        - that path does **not** pass through any critical-chain task before `C`.
    - To find all such `t`, iterate over all tasks `t` not on the critical chain and check:

        - is `C` in `reachability[t]`? If yes, find the first critical-chain node encountered on any path from `t` to the critical chain — if that first node is `C`, `t` feeds `C`.

## 4. **Group feeding tasks into feeding chains**

    - For each feeding endpoint `t`, trace the unique (or all) path(s) from `t` to `C` but stop if you encounter another feeding endpoint — you want to identify distinct chains. In practice:

        - For DAGs, you can find the **set of nodes reachable from `t` that do not include other critical nodes** and that lead to `C`.
        - Represent each feeding chain as the set of tasks that are on a path from the chain start to `C` without touching the critical chain earlier.

## 5. **Detect merges between feeding chains**

    - Two feeding chains merge before `C` if their node-sets intersect or if there exists a node `m` that is reachable from both starts and from `m` there is still a path to `C` without hitting the critical chain.
    - Merge logic:

        - Build a directed acyclic subgraph consisting of all nodes that reach `C` without encountering the critical chain earlier.
        - Compute weakly-connected components in that subgraph *after* ignoring critical-chain nodes — each connected component either feeds independently or merges into a single feed.
        - For each component, determine the boundary node that immediately precedes the critical chain (i.e., nodes whose successors include a critical-chain node or nodes that feed into the node that does). That boundary is where the feeding buffer will be placed.

## 6. **Buffer sizing**

    - Standard CCPM rule: buffer ≈ 50% of the total safety removed from that feeding chain.
    - Calculation:

        - For each feeding chain, sum removed safety = Σ(original\_duration − aggressive\_duration) for tasks in that chain.
        - Set `feeding_buffer_size = alpha * removed_safety` where `alpha` is typically 0.5 (or parameterizable).
    - Place the buffer **after** the last feeding-chain task and **before** the critical-chain task `C`.

## 7. **Edge cases**

    - If feeding chains fully overlap (one is subset of another) — treat as a single feeding chain.
    - If a feeding chain merges into another chain *and then splits again* before `C` — collapse the merge region into one feeding buffer if the split is inside the merged component and both branches ultimately converge before `C`.
    - If resources are shared within feeding chains, consider whether merging reduces or increases buffer needs — often still use the same buffer calculus (based on removed safety).

## 8. **Annotate & return**

    - For each buffer: `placed_after_task` is the node just before the critical chain (or the merged node if multiple merge).
    - Record which feeding-task IDs the buffer protects.
    - Optionally annotate tasks with `feeds_buffer_id` for monitoring/reporting.


# **Differences between buffer types**

## **Project Buffer**

- **Purpose:** Protects the *entire project* from delays in the critical chain.
- **Placement:** End of the critical chain, just before project completion milestone.
- **Sizing:** ~50% of total removed safety from *critical chain* tasks.
- **Monitored Against:** Final delivery date.
- **Impact of Erosion:** Directly reduces likelihood of on-time completion.

## **Feeding Buffer**

- **Purpose:** Protects the critical chain from delays in *non-critical* feeding chains.
- **Placement:** Between the end of a feeding chain and the first task it feeds on the critical chain.
- **Sizing:** ~50% of removed safety from that feeding chain’s tasks.
- **Monitored Against:** Start time of the merge point on the critical chain.
- **Impact of Erosion:** Risk of causing delays to the critical chain (indirect project impact).


# Diagnostics

one unified reporting system so whether the blocker is:

- a missing predecessor,
- a predecessor not scheduled yet,
- a circular dependency, or
- a resource/capacity issue,

…it will all come from one consistent check that can be run before or during scheduling.

## Benefits

- You can now **iterate toward a valid schedule** without guessing — it will tell you:

    - *exactly* which dependency is missing
    - if there’s a **cycle**, with the **entire chain printed**
    - whether a resource calendar or capacity is the issue
- Diagnostic output stays in one place, so any other scheduling mode can reuse it.
