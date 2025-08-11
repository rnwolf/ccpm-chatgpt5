# ccpm_module.py
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd


# ------------------------
# Domain classes (unchanged)
# ------------------------
class ProjectCalendar:
    """
    Project-wide calendar for defining global non-working days.
    Days are integer indices (0,1,2,...).
    """

    def __init__(self, non_working_days: Optional[List[int]] = None):
        self.non_working_days = set(non_working_days or [])

    def is_working_day(self, day_index: int) -> bool:
        return day_index not in self.non_working_days

    def to_dict(self) -> Dict:
        return {"non_working_days": sorted(self.non_working_days)}

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(non_working_days=data.get("non_working_days", []))


class Resource:
    """
    Represents a named resource and its non-working days.
    """

    def __init__(
        self, resource_id: str, name: str, non_working_days: Optional[List[int]] = None
    ):
        self.id = resource_id
        self.name = name
        self.non_working_days = set(non_working_days or [])

    def is_available(self, day_index: int, project_calendar: ProjectCalendar) -> bool:
        return (
            project_calendar.is_working_day(day_index)
            and day_index not in self.non_working_days
        )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "non_working_days": sorted(self.non_working_days),
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            resource_id=data["id"],
            name=data["name"],
            non_working_days=data.get("non_working_days", []),
        )


class Task:
    """
    Represents a project task in CCPM.
    """

    def __init__(
        self,
        task_id: str,
        name: str,
        duration: int,
        resources: Optional[List[str]] = None,
        predecessors: Optional[List[str]] = None,
        description: str = "",
    ):
        self.id = task_id
        self.name = name
        self.duration = duration
        self.resources = resources or []  # list of resource IDs
        self.predecessors = predecessors or []
        self.description = description
        # schedule fields
        self.slack = 0
        self.asap_start: Optional[int] = None
        self.asap_finish: Optional[int] = None
        self.alap_start: Optional[int] = None
        self.alap_finish: Optional[int] = None
        self.scheduled_start: Optional[int] = None
        self.scheduled_finish: Optional[int] = None
        self.on_critical_chain: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "resources": ",".join(self.resources),
            "predecessors": ",".join(self.predecessors),
            "description": self.description,
            "asap_start": self.asap_start,
            "asap_finish": self.asap_finish,
            "alap_start": self.alap_start,
            "alap_finish": self.alap_finish,
            "scheduled_start": self.scheduled_start,
            "scheduled_finish": self.scheduled_finish,
            "on_critical_chain": self.on_critical_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        resources = [
            r.strip() for r in str(data.get("resources", "")).split(",") if r.strip()
        ]
        preds = [
            p.strip() for p in str(data.get("predecessors", "")).split(",") if p.strip()
        ]
        t = cls(
            task_id=data["id"],
            name=data.get("name", data["id"]),
            duration=int(data["duration"]),
            resources=resources,
            predecessors=preds,
            description=data.get("description", ""),
        )
        t.asap_start = data.get("asap_start")
        t.asap_finish = data.get("asap_finish")
        t.alap_start = data.get("alap_start")
        t.alap_finish = data.get("alap_finish")
        t.scheduled_start = data.get("scheduled_start")
        t.scheduled_finish = data.get("scheduled_finish")
        t.on_critical_chain = bool(data.get("on_critical_chain", False))
        return t


class Buffer(Task):
    def __init__(
        self,
        buffer_id: str,
        name: str,
        duration: int,
        linked_task: Optional[str] = None,
        description: str = "",
    ):
        super().__init__(
            buffer_id,
            name,
            duration,
            resources=[],
            predecessors=[],
            description=description,
        )
        self.linked_task = linked_task

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["linked_task"] = self.linked_task
        return d

    @classmethod
    def from_dict(cls, data: Dict):
        buf = cls(
            buffer_id=data["id"],
            name=data.get("name", data["id"]),
            duration=int(data["duration"]),
            linked_task=data.get("linked_task"),
            description=data.get("description", ""),
        )
        buf.scheduled_start = data.get("scheduled_start")
        buf.scheduled_finish = data.get("scheduled_finish")
        return buf


class ProjectBuffer(Buffer):
    def __init__(
        self,
        buffer_id: str,
        name: str,
        duration: int,
        delivery_date: Optional[int] = None,
        description: str = "",
    ):
        super().__init__(
            buffer_id, name, duration, linked_task=None, description=description
        )
        self.delivery_date = delivery_date

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["delivery_date"] = self.delivery_date
        return d

    @classmethod
    def from_dict(cls, data: Dict):
        buf = cls(
            buffer_id=data["id"],
            name=data.get("name", data["id"]),
            duration=int(data["duration"]),
            delivery_date=data.get("delivery_date"),
            description=data.get("description", ""),
        )
        buf.scheduled_start = data.get("scheduled_start")
        buf.scheduled_finish = data.get("scheduled_finish")
        return buf


class FeedingBuffer(Buffer):
    pass


# ------------------------
# Helpers: working-day arithmetic
# ------------------------
def next_working_day(start: int, working_days: Set[int]) -> int:
    d = start
    while d not in working_days:
        d += 1
    return d


def prev_working_day(start: int, working_days: Set[int]) -> int:
    d = start
    while d not in working_days:
        d -= 1
    return d


def add_working_days_forward(start: int, days: int, working_days: Set[int]) -> int:
    """Return the last day index (inclusive) after adding 'days' working days starting at 'start' inclusive.
    Example: start=2, days=1 --> returns 2 (task of duration 1 occupies day 2).
    """
    if days <= 0:
        return start - 1
    d = start
    remaining = days
    while remaining > 0:
        if d in working_days:
            remaining -= 1
            if remaining == 0:
                return d
        d += 1
    return d - 1


def subtract_working_days_backward(
    end_day: int, duration: int, project_calendar: ProjectCalendar, resource: Resource
) -> int:
    """Count backwards from end_day to find the earliest start day
    given duration in working days, skipping non-working days."""
    remaining = duration
    day = end_day
    while remaining > 0:
        day -= 1
        if project_calendar.is_working_day(day) and resource.is_available(
            day, project_calendar
        ):
            remaining -= 1
    return day


# ------------------------
# Graph helpers
# ------------------------
def compute_successors(tasks: Dict[str, Task]) -> Dict[str, List[str]]:
    succs: Dict[str, List[str]] = {}
    for tid, t in tasks.items():
        for p in t.predecessors:
            succs.setdefault(p, []).append(tid)
    return succs


def topological_sort(tasks: Dict[str, Task]) -> List[str]:
    indeg = {tid: 0 for tid in tasks}
    for t in tasks.values():
        for p in t.predecessors:
            indeg[t.id] += 1
    q = [tid for tid, deg in indeg.items() if deg == 0]
    order = []
    while q:
        n = q.pop(0)
        order.append(n)
        for succ in [tid for tid, tt in tasks.items() if n in tt.predecessors]:
            indeg[succ] -= 1
            if indeg[succ] == 0:
                q.append(succ)
    if len(order) != len(tasks):
        raise RuntimeError("Cycle detected in task graph")
    return order


def detect_cycles(tasks: list) -> None:
    """
    Check for circular dependencies and raise an error if found.
    Also checks for dependencies on missing tasks.
    """
    task_map = {t.id: t for t in tasks}
    visited = {}
    stack = []

    def visit(task_id):
        if visited.get(task_id) == "visiting":
            cycle_start_index = stack.index(task_id)
            cycle = stack[cycle_start_index:] + [task_id]
            raise RuntimeError(f"Circular dependency detected: {' -> '.join(cycle)}")
        if visited.get(task_id) == "done":
            return
        visited[task_id] = "visiting"
        stack.append(task_id)
        for pred in task_map[task_id].predecessors:
            if pred not in task_map:
                raise RuntimeError(f"Task {task_id} depends on unknown task {pred}")
            visit(pred)
        stack.pop()
        visited[task_id] = "done"

    for t in tasks:
        visit(t.id)


def diagnose_unschedulable(tasks, resources, project_calendar, sched_map=None):
    """
    Returns a dict: { task_id: [list of reason strings] }
    Covers missing preds, unscheduled preds, circular deps, resource issues.
    """
    task_ids = {t.id for t in tasks}
    reasons_map = {t.id: [] for t in tasks}

    # --- Detect cycles first ---
    cycles = find_cycles(tasks)
    for cycle in cycles:
        for tid in cycle:
            reasons_map[tid].append(
                f"Circular dependency in chain: {' -> '.join(cycle)}"
            )

    for t in tasks:
        # --- Dependency checks ---
        missing_preds = [pred for pred in t.predecessors if pred not in task_ids]
        if missing_preds:
            reasons_map[t.id].append(
                f"Missing predecessor(s): {', '.join(missing_preds)}"
            )

        unscheduled_preds = [
            pred
            for pred in t.predecessors
            if sched_map is None or pred not in sched_map
        ]
        if unscheduled_preds:
            reasons_map[t.id].append(
                f"Predecessor(s) not yet scheduled: {', '.join(unscheduled_preds)}"
            )

        # --- Resource checks ---
        task_res = next((r for r in resources if r.id == t.resource_id), None)
        if not task_res:
            reasons_map[t.id].append("Resource not found")
        else:
            if len(task_res.available_days) == 0:
                reasons_map[t.id].append("Resource has no available days")
            else:
                overlap_days = [
                    d
                    for d in task_res.available_days
                    if d in project_calendar.working_days
                ]
                if not overlap_days:
                    reasons_map[t.id].append(
                        "No overlap between resource calendar and project calendar"
                    )

    # Filter out tasks with no reasons
    return {tid: reasons for tid, reasons in reasons_map.items() if reasons}


def find_cycles(tasks):
    """
    Returns list of cycles, where each cycle is a list of task IDs in order.
    Uses DFS to detect and record cycles.
    """
    graph = {t.id: t.predecessors for t in tasks}
    visited = set()
    stack = []
    cycles = []

    def dfs(node, path):
        if node in path:
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        for pred in graph.get(node, []):
            dfs(pred, path + [node])

    for tid in graph:
        dfs(tid, [])
    return cycles


# ------------------------
# ASAP forward pass (ignores resources)
# ------------------------
def compute_asap(tasks: Dict[str, Task], project_calendar: ProjectCalendar):
    """Compute ASAP start/finish for each task (earliest), ignoring resource constraints."""
    working_days = set(
        d
        for d in range(max(project_calendar.non_working_days or [0]) + 365)
        if project_calendar.is_working_day(d)
    )
    order = topological_sort(tasks)
    for tid in order:
        t = tasks[tid]
        if not t.predecessors:
            # earliest working day >= 0
            start = next_working_day(0, working_days)
        else:
            # earliest start is the day after the max asap_finish of predecessors
            start = max(tasks[p].asap_finish + 1 for p in t.predecessors)
            start = next_working_day(start, working_days)
        finish = add_working_days_forward(start, t.duration, working_days)
        t.asap_start = start
        t.asap_finish = finish


# ------------------------
# ALAP backward pass (ignores resources)
# ------------------------
def compute_alap(
    tasks: Dict[str, Task], project_calendar: ProjectCalendar, delivery_day: int
):
    """Compute ALAP start/finish anchored at delivery_day (latest possible ignoring resources)."""
    working_days = set(
        d
        for d in range(max(project_calendar.non_working_days or [0]) + 365)
        if project_calendar.is_working_day(d)
    )
    # ensure delivery_day is a working day, otherwise move to previous working day
    if delivery_day not in working_days:
        delivery_day = prev_working_day(delivery_day, working_days)
    succs = compute_successors(tasks)
    order = list(reversed(topological_sort(tasks)))
    for tid in order:
        t = tasks[tid]
        if not succs.get(tid):
            # terminal tasks end by delivery_day
            finish = delivery_day
        else:
            # finish = min(start of successors) - 1 (but must be working day)
            finish = min(tasks[s].alap_start - 1 for s in succs[tid])
            if finish not in working_days:
                finish = prev_working_day(finish, working_days)
        # compute start by stepping backwards duration working days
        # find earliest day d such that adding duration working days from d gives finish
        # we can compute start by walking backwards
        remaining = t.duration
        d = finish
        while remaining > 0:
            if d in working_days:
                remaining -= 1
                if remaining == 0:
                    start = d
                    break
            d -= 1
        t.alap_start = start
        t.alap_finish = finish


# ------------------------
# Resource-constrained ALAP list scheduling
# ------------------------
def resource_constrained_alap(
    tasks: Dict[str, Task],
    resources: Dict[str, Resource],
    project_calendar: ProjectCalendar,
    max_day: int = 365,
) -> Dict[str, Tuple[int, int]]:
    """
    Try to place tasks within [alap_start..alap_finish] windows while respecting resource availability.
    This places tasks as late as possible (ALAP placement) to protect project buffer.
    Returns dict of scheduled starts/finishes.
    Raises RuntimeError if can't place.
    """
    # Prepare resource daily allocation map (0/1 availability per resource per day)
    allocated: Dict[str, Dict[int, int]] = {
        rid: {d: 0 for d in range(max_day + 1)} for rid in resources
    }
    working_days = set(
        d for d in range(max_day + 1) if project_calendar.is_working_day(d)
    )

    # Priority: tasks with smallest slack (alap_start - asap_start) first; critical tasks first
    for t in tasks.values():
        if t.asap_start is None:
            t.asap_start = 0
            t.asap_finish = 0
        t.slack = (
            (t.alap_start - t.asap_start)
            if (t.alap_start is not None and t.asap_start is not None)
            else 0
        )
        t.on_critical_chain = t.slack == 0

    # build list sorted: critical tasks first, then by increasing slack, then by later ALAP (so late placement)
    ordered = sorted(
        tasks.values(),
        key=lambda x: (0 if x.on_critical_chain else 1, x.slack, -(x.alap_start or 0)),
    )

    scheduled_count = 0
    scheduled = set()
    # We'll iterate until we either schedule all or make no progress in a full pass
    progress = True
    while scheduled_count < len(tasks) and progress:
        progress = False
        for t in ordered:
            if t.id in scheduled:
                continue
            # compute latest allowed start (alap_start)
            latest_start = t.alap_start
            if latest_start is None:
                latest_start = max_day - t.duration + 1
            # earliest allowed start maybe after predecessors scheduled finish +1
            if t.predecessors:
                if any(tasks[p].scheduled_finish is None for p in t.predecessors):
                    # predecessors not scheduled yet: cannot schedule this task now
                    continue
                earliest_allowed = max(
                    tasks[p].scheduled_finish + 1 for p in t.predecessors
                )
            else:
                earliest_allowed = 0
            # clamp latest_start to be >= earliest_allowed
            if latest_start < earliest_allowed:
                # cannot place within ALAP window
                raise RuntimeError(
                    f"Task {t.id} cannot be placed: ALAP window ends before predecessors finish (latest_start {latest_start} < earliest_allowed {earliest_allowed})."
                )
            # try to place at the latest possible start (move backwards)
            placed = False
            start_candidate = latest_start
            while start_candidate >= earliest_allowed:
                # compute sequence of days this task would occupy (working days)
                # find first working day >= start_candidate (if start_candidate not working move forward)
                # then check if there are exactly t.duration working days starting from there that end within <= latest_start_end
                # Simpler: attempt to allocate contiguous calendar days but only count days that are working
                # We'll consider the candidate start as a working-day start.
                if start_candidate not in working_days:
                    start_candidate = prev_working_day(start_candidate, working_days)
                # compute end day given this start
                end_day = add_working_days_forward(
                    start_candidate, t.duration, working_days
                )
                # check that end_day <= t.alap_finish (we can't exceed ALAP finish)
                if t.alap_finish is not None and end_day > t.alap_finish:
                    # start too late; move earlier
                    start_candidate -= 1
                    continue
                # check resource availability for each day in the working days in [start_candidate..end_day]
                # build list of days used
                days_used = [
                    d for d in range(start_candidate, end_day + 1) if d in working_days
                ]
                ok = True
                for r_id in t.resources:
                    if r_id not in resources:
                        ok = False
                        break
                    for d in days_used:
                        if not resources[r_id].is_available(d, project_calendar):
                            ok = False
                            break
                        if allocated[r_id].get(d, 0) + 1 > 1:
                            # here we treat resources as single-capacity; if you need capacities >1 extend Resource
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # allocate
                    for r_id in t.resources:
                        for d in days_used:
                            allocated[r_id][d] += 1
                    t.scheduled_start = start_candidate
                    t.scheduled_finish = end_day
                    scheduled.add(t.id)
                    scheduled_count += 1
                    progress = True
                    placed = True
                    break
                # else try earlier start
                start_candidate -= 1
            if not placed:
                # cannot place this task in this pass - skip for now (maybe when predecessors place later it will be possible)
                continue
        # end for ordered
        if not progress and scheduled_count < len(tasks):
            raise RuntimeError(
                "No progress in resource constrained scheduling pass; tasks cannot be scheduled with current constraints (possible missing capacity or circular dependencies)."
            )

    # Filter out tasks with None values or raise an error
    result = {}
    for t in tasks.values():
        if t.scheduled_start is None or t.scheduled_finish is None:
            raise ValueError(f"Task {t.id} is not fully scheduled")
        result[t.id] = (t.scheduled_start, t.scheduled_finish)
    return result


# ------------------------
# schedule_tasks wrapper: full flow (ASAP -> ALAP -> resource-constrained ALAP)
# ------------------------
def schedule_tasks(
    tasks, resources, project_calendar, max_day=60, max_iters=6, diagnostic=True
):
    detect_cycles(tasks)  # hard fail for cycles if needed

    for iteration in range(max_iters):
        sched_map = resource_constrained_alap(
            tasks, resources, project_calendar, max_day=max_day
        )
        if sched_map is not None:
            return sched_map

        if diagnostic:
            print(f"\n[DIAGNOSTIC MODE] Iteration {iteration+1}: Unschedulable tasks:")
            reasons = diagnose_unschedulable(
                tasks, resources, project_calendar, sched_map=sched_map
            )
            for tid, why in reasons.items():
                print(f"  Task {tid}: {', '.join(why)}")

        raise RuntimeError(
            "No progress in scheduling; see diagnostics for unschedulable tasks."
        )


# ------------------------
# DataFrame helpers (your existing helpers kept)
# ------------------------
def tasks_to_df(tasks: List[Task]) -> pd.DataFrame:
    rows = []
    for t in tasks:
        rows.append(
            {
                "id": t.id,
                "name": t.name,
                "duration": t.duration,
                "resources": ",".join(t.resources),
                "predecessors": ",".join(t.predecessors),
                "description": t.description,
                "asap_start": t.asap_start,
                "asap_finish": t.asap_finish,
                "alap_start": t.alap_start,
                "alap_finish": t.alap_finish,
                "scheduled_start": t.scheduled_start,
                "scheduled_finish": t.scheduled_finish,
            }
        )
    return pd.DataFrame(rows)


def df_to_tasks(df: pd.DataFrame) -> List[Task]:
    tasks = []
    for _, row in df.iterrows():
        resources = [
            r.strip() for r in str(row.get("resources", "")).split(",") if r.strip()
        ]
        preds = [
            p.strip() for p in str(row.get("predecessors", "")).split(",") if p.strip()
        ]
        t = Task(
            task_id=row["id"],
            name=row.get("name", row["id"]),
            duration=int(row["duration"]),
            resources=resources,
            predecessors=preds,
            description=row.get("description", ""),
        )
        t.asap_start = row.get("asap_start")
        t.asap_finish = row.get("asap_finish")
        t.alap_start = row.get("alap_start")
        t.alap_finish = row.get("alap_finish")
        t.scheduled_start = row.get("scheduled_start")
        t.scheduled_finish = row.get("scheduled_finish")
        tasks.append(t)
    return tasks


def resources_to_availability_df(
    resources: List[Resource], max_day: int = 31
) -> pd.DataFrame:
    data = {}
    id_to_name = {}
    for res in resources:
        col = []
        for day in range(max_day):
            col.append(1 if day in res.non_working_days else 0)
        data[res.id] = col
        id_to_name[res.id] = res.name
    df = pd.DataFrame(data, index=range(max_day))
    df.index.name = "Day"
    df.attrs["id_to_name"] = id_to_name
    return df


def resources_df_to_availability(df: pd.DataFrame) -> List[Resource]:
    resources = []
    id_to_name = df.attrs.get("id_to_name", {})
    for col in df.columns:
        non_working_days = df.index[df[col] == 1].tolist()
        name = id_to_name.get(col, col)
        resources.append(
            Resource(resource_id=col, name=name, non_working_days=non_working_days)
        )
    return resources


# ------------------------
# Example usage / smoke test
# ------------------------
if __name__ == "__main__":
    # project calendar: weekends days 5,6,12,13 (two weekends shown)
    # calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    calendar = ProjectCalendar(
        non_working_days=[d for d in range(0, 100) if d % 7 in (5, 6)]
    )

    resources = [
        Resource("R1", "Alice", non_working_days=[]),  # Alice off on day 7
        Resource("R2", "Bob", non_working_days=[]),
        Resource("R3", "Charlie", non_working_days=[]),
    ]

    tasks = [
        Task("T1", "Spec", 4, resources=["R1"], predecessors=[]),
        Task("T2", "Develop", 6, resources=["R1", "R2"], predecessors=["T1"]),
        Task("T3", "Test", 3, resources=["R2"], predecessors=["T2"]),
        Task("T4", "Doc", 2, resources=["R1"], predecessors=["T1"]),
    ]

    schedule_map = schedule_tasks(tasks, resources, calendar, max_day=60, max_iters=6)
    print("Schedule results:")
    for tid, (s, e) in schedule_map.items():
        print(tid, "start:", s, "finish:", e)
    # For viewing in marimo, you can convert tasks to DataFrame
    df = tasks_to_df(tasks)
    print(df.to_string(index=False))
