# ccpm_module3.py
from typing import List, Dict, Optional, Set, Tuple, Union
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

    def next_working_day(self, day: int) -> int:
        d = day
        while not self.is_working_day(d):
            d += 1
        return d

    def add_working_days(self, start: int, days: int) -> int:
        """
        Returns the day index *after* the task finishes.
        """
        if days <= 0:
            return start

        d = start
        remaining = days
        while remaining > 0:
            if self.is_working_day(d):
                remaining -= 1
            if remaining > 0:
                d += 1
        return d + 1

    def prev_working_day(self, day: int) -> int:
        d = day
        while not self.is_working_day(d):
            d -= 1
        return d

    def subtract_working_days(self, end_day: int, days: int) -> int:
        """
        Returns the start day index given an end day and duration.
        A task of duration 1 ending on a working day `d` will have a start day of `d`.
        """
        if days <= 0:
            return end_day

        d = end_day
        remaining = days
        while remaining > 0:
            if self.is_working_day(d):
                remaining -= 1
            if remaining == 0:
                break
            d -= 1
        return d

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
        self, resource_id: str, name: str, non_working_days: Optional[List[int]] = None, capacity_per_day: int = 1
    ):
        self.id = resource_id
        self.name = name
        self.non_working_days = set(non_working_days or [])
        self.capacity_per_day = capacity_per_day
        self.allocations: Dict[int, int] = {}

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
            "capacity_per_day": self.capacity_per_day,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            resource_id=data["id"],
            name=data["name"],
            non_working_days=data.get("non_working_days", []),
            capacity_per_day=data.get("capacity_per_day", 1),
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
        self.successors: List[str] = []

    def get_predecessors(self) -> List[str]:
        """Returns a list of predecessor task IDs."""
        return self.predecessors

    def get_successors(self) -> List[str]:
        """Returns a list of successor task IDs."""
        return self.successors

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


def diagnose_unschedulable(tasks, resources, project_calendar, forward_map):
    """
    Prints detailed reasons why each remaining task can't be scheduled.
    """
    for tid, task in tasks.items():
        reasons = []

        # Predecessor blocking
        for pid in task.predecessors:
            if pid not in forward_map:
                reasons.append(f"predecessor '{pid}' is not scheduled")
            else:
                pred_end = forward_map[pid][1]
                if pred_end > task.latest_finish:
                    reasons.append(
                        f"predecessor '{pid}' finishes at day {pred_end}, "
                        f"after task's latest finish {task.latest_finish}"
                    )

        # Resource availability blocking
        resource = resources[task.resource_id]
        if not project_calendar.is_resource_available_for_span(
            resource, task.latest_finish - task.duration + 1, task.latest_finish
        ):
            reasons.append(f"resource '{resource.name}' unavailable in latest slot")

        if reasons:
            print(f"Task '{tid}' is unschedulable because:")
            for r in reasons:
                print(f"  - {r}")


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


def detect_circular_dependencies(tasks: dict[str, "Task"]) -> list[list[str]]:
    """
    Detects circular dependencies in the task graph.
    Returns a list of cycles, where each cycle is a list of task IDs.
    """
    visited = set()
    stack = set()
    cycles = []
    path = []

    def visit(tid: str):
        if tid in stack:
            # Found a cycle — extract the cycle path
            if tid in path:
                idx = path.index(tid)
                cycles.append(path[idx:] + [tid])
            return
        if tid in visited:
            return

        visited.add(tid)
        stack.add(tid)
        path.append(tid)
        for pid in tasks[tid].predecessors:
            if pid in tasks:  # Ignore missing refs here; they'll be caught elsewhere
                visit(pid)
        stack.remove(tid)
        path.pop()

    for tid in tasks.keys():
        visit(tid)

    return cycles


# ------------------------
# ASAP forward pass (ignores resources)
# ------------------------
def compute_asap(
    tasks: Dict[str, Task], project_calendar: ProjectCalendar, max_day: int = 365
) -> Dict[str, Tuple[int, int]]:
    """
    Compute ASAP (earliest) start/finish for each task (ignoring resource constraints).
    - tasks: dict keyed by task id
    - project_calendar: ProjectCalendar
    - max_day: horizon to build working-day set
    Returns a mapping {task_id: (asap_start, asap_finish)} and also populates task.asap_start/asap_finish.
    """
    order = topological_sort(tasks)  # expects tasks is a dict
    for tid in order:
        t = tasks[tid]
        if not t.predecessors:
            start = project_calendar.next_working_day(0)
        else:
            # predecessor asap_finish must exist — if not, treat as 0 (you can refine this)
            start = max((tasks[p].asap_finish or 0) + 1 for p in t.predecessors)
            start = project_calendar.next_working_day(start)
        # add_working_days returns the day AFTER, so subtract 1 for inclusive finish day
        finish = project_calendar.add_working_days(start, t.duration) - 1
        t.asap_start = start
        t.asap_finish = finish

    return {tid: (t.asap_start, t.asap_finish) for tid, t in tasks.items()}


# ------------------------
# ALAP backward pass (ignores resources)
# ------------------------
def compute_alap(
    tasks: Dict[str, Task], project_calendar: ProjectCalendar, delivery_day: int
):
    """Compute ALAP start/finish anchored at delivery_day (latest possible ignoring resources)."""
    delivery_day = project_calendar.prev_working_day(delivery_day)

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
            finish = project_calendar.prev_working_day(finish)

        # compute start by stepping backwards duration working days
        start = project_calendar.subtract_working_days(finish, t.duration)
        t.alap_start = start
        t.alap_finish = finish


# ------------------------
# Resource-constrained ALAP list scheduling
# ------------------------
def resource_constrained_alap(
    tasks: Dict[str, Task],
    resources: Dict[str, Resource],
    project_calendar: ProjectCalendar,
    max_day: int,
    diagnostic: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """
    Resource-constrained As Late As Possible (ALAP) scheduling.
    Returns dict: task_id -> (start_day, finish_day)
    """
    # Normalize tasks into a dict
    if isinstance(tasks, list):
        task_lookup = {t.id: t for t in tasks}
    else:
        task_lookup = tasks

    # --- Detect circular dependencies first ---
    visited, rec_stack = set(), set()

    def dfs(task_id: str) -> bool:
        visited.add(task_id)
        rec_stack.add(task_id)
        for pred_id in tasks[task_id].predecessors:
            if pred_id not in tasks:
                continue
            if pred_id not in visited and dfs(pred_id):
                return True
            elif pred_id in rec_stack:
                raise RuntimeError(f"Circular dependency detected involving {pred_id}")
        rec_stack.remove(task_id)
        return False

    for tid in tasks:
        if tid not in visited:
            dfs(tid)

    # --- Initialize scheduling map ---
    schedule: Dict[str, Tuple[int, int]] = {tid: (None, None) for tid in tasks}

    # Start with all tasks unscheduled
    unscheduled = set(task_lookup.keys())
    order = list(reversed(topological_sort(task_lookup)))
    progress = True
    iteration = 0

    while unscheduled and progress:
        progress = False
        iteration += 1

        for tid in order:
            if tid not in unscheduled:
                continue

            task = task_lookup[tid]

            # Check successors scheduled? (This is a backward pass)
            succs_done = all(
                schedule[succ_id][0] is not None for succ_id in task.successors
            )

            if not succs_done:
                if diagnostic:
                    missing = [s for s in task.successors if schedule[s][0] is None]
                    print(f"Task {tid} blocked: unscheduled successors {missing}")
                continue

            # Determine latest possible finish = min(start of successors) - 1
            if task.successors:
                min_succ_start = min(
                    schedule[sid][0]
                    for sid in task.successors
                    if schedule[sid][0] is not None
                )
                latest_finish = min_succ_start - 1
            else:
                latest_finish = max_day

            # Duration in working days
            duration_days = project_calendar.add_working_days(0, task.duration) - 0

            # Try scheduling as late as possible without breaking resource limits
            scheduled = False
            for start_day in range(latest_finish - duration_days, -1, -1):
                finish_day = (
                    project_calendar.add_working_days(start_day, task.duration) - 1
                )

                # Check resource availability for ALL resources in task
                resources_ok = True
                for res_id in task.resources:
                    res = resources[res_id]
                    units_required = 1  # Assuming 1 unit per resource
                    if any(
                        res.allocations.get(day, 0) + units_required
                        > res.capacity_per_day
                        for day in range(start_day, finish_day + 1)
                        if project_calendar.is_working_day(day)
                    ):
                        resources_ok = False
                        if diagnostic:
                            print(
                                f"Task {tid} blocked at {start_day}-{finish_day} by resource {res_id} (capacity exceeded)"
                            )
                        break
                if not resources_ok:
                    continue

                if resources_ok:
                    # Allocate resources
                    for res_id in task.resources:
                        res = resources[res_id]
                        units_required = 1  # Assuming 1 unit per resource
                        for day in range(start_day, finish_day + 1):
                            if project_calendar.is_working_day(day):
                                res.allocations[day] = (
                                    res.allocations.get(day, 0) + units_required
                                )

                    schedule[tid] = (start_day, finish_day)
                    unscheduled.remove(tid)
                    scheduled = True
                    progress = True
                    break

            if not scheduled and diagnostic:
                print(f"Task {tid} could not be scheduled (no valid slot found)")

    if unscheduled:
        raise RuntimeError(
            f"No progress in resource constrained ALAP; tasks unscheduled: {unscheduled}"
        )

    return schedule


# ------------------------
# schedule_tasks wrapper: full flow (ASAP -> ALAP -> resource-constrained ALAP)
# ------------------------
def schedule_tasks(
    tasks: Union[List[Task], Dict[str, Task]],
    resources: Dict[str, Resource],
    project_calendar: ProjectCalendar,
    max_day: int,
    max_iters: int = 10,
    diagnostic: bool = False,
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Perform scheduling using both ASAP and resource-constrained ALAP passes.
    Supports tasks as a list of Task objects or dict keyed by task ID.
    """
    # Normalize tasks into a dict
    if isinstance(tasks, list):
        task_lookup = {t.id: t for t in tasks}
    else:
        task_lookup = tasks

    # Compute and assign successors
    successors_map = compute_successors(task_lookup)
    for tid, task in task_lookup.items():
        task.successors = successors_map.get(tid, [])

    if diagnostic:
        print("\n=== FORWARD PASS (ASAP) ===")

    # --- Forward pass (ASAP) ---
    asap_schedule: Dict[str, Tuple[int, int]] = {}
    unscheduled = set(task_lookup.keys())
    iteration = 0
    progress = True

    while unscheduled and progress and iteration < max_iters:
        iteration += 1
        progress = False

        for tid in list(unscheduled):
            task = task_lookup[tid]

            # Predecessors must be scheduled first
            if not all(t in asap_schedule for t in task.predecessors):
                if diagnostic:
                    missing = [p for p in task.predecessors if p not in asap_schedule]
                    print(f"ASAP: Task {tid} blocked by predecessors {missing}")
                continue

            # Earliest start = max of predecessors' finish days + 1
            earliest_start = 0
            if task.predecessors:
                earliest_start = max(asap_schedule[p][1] for p in task.predecessors) + 1

            # Schedule ASAP (no resource check for this pass)
            start_day = project_calendar.next_working_day(earliest_start)
            finish_day = project_calendar.add_working_days(start_day, task.duration) - 1

            asap_schedule[tid] = (start_day, finish_day)
            unscheduled.remove(tid)
            progress = True

            if diagnostic:
                print(f"ASAP: Scheduled Task {tid} {start_day}-{finish_day}")

    if unscheduled:
        raise RuntimeError(f"ASAP scheduling stuck; unscheduled tasks: {unscheduled}")

    # --- Backward pass with resources (ALAP) ---
    if diagnostic:
        print("\n=== BACKWARD PASS (ALAP) ===")

    alap_schedule = resource_constrained_alap(
        task_lookup,
        resources,
        project_calendar,
        max_day=max_day,
        diagnostic=diagnostic,
    )

    # --- Merge results and update task objects ---
    for tid, task in task_lookup.items():
        if tid in asap_schedule:
            task.asap_start, task.asap_finish = asap_schedule[tid]
        if tid in alap_schedule:
            task.alap_start, task.alap_finish = alap_schedule[tid]
            task.scheduled_start, task.scheduled_finish = alap_schedule[tid]

    schedule_map = {
        "ASAP": asap_schedule,
        "ALAP": alap_schedule,
    }

    if diagnostic:
        print("\n=== FINAL SCHEDULE ===")
        for tid in task_lookup:
            asap_s = schedule_map["ASAP"][tid]
            alap_s = schedule_map["ALAP"][tid]
            print(f"Task {tid}: ASAP {asap_s}, ALAP {alap_s}")

    return schedule_map


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

    resources_list = [
        Resource("R1", "Alice", non_working_days=[]),
        Resource("R2", "Bob", non_working_days=[]),
        Resource("R3", "Charlie", non_working_days=[]),
    ]
    resources_map = {r.id: r for r in resources_list}


    tasks = [
        Task("T1", "Spec", 4, resources=["R1"], predecessors=[]),
        Task("T2", "Develop", 6, resources=["R1", "R2"], predecessors=["T1"]),
        Task("T3", "Test", 3, resources=["R2"], predecessors=["T2"]),
        Task("T4", "Doc", 2, resources=["R1"], predecessors=["T1"]),
    ]

    schedule_map = schedule_tasks(tasks, resources_map, calendar, max_day=60, max_iters=10)
    print("Schedule results:")
    for schedule_name, schedule in schedule_map.items():
        print(f"--- {schedule_name} ---")
        for tid, (s, e) in sorted(schedule.items()):
            print(f"  Task {tid}: start: {s}, finish: {e}")

    # For viewing in marimo, you can convert tasks to DataFrame
    df = tasks_to_df(tasks)
    print("\n--- Task Details ---")
    print(df.to_string(index=False))
