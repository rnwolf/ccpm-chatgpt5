# ccpm_module3.py
from typing import List, Dict, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class TaskStatus(Enum):
    """Execution status of a task."""
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"


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

    def to_dict(self) -> Dict[str, Any]:
        return {"non_working_days": sorted(list(self.non_working_days))}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectCalendar":
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "non_working_days": sorted(list(self.non_working_days)),
            "capacity_per_day": self.capacity_per_day,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Resource":
        return cls(
            resource_id=data["id"],
            name=data["name"],
            non_working_days=data.get("non_working_days", []),
            capacity_per_day=data.get("capacity_per_day", 1),
        )


class Task:
    """
    Represents a project task in CCPM, enhanced with execution tracking.
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
        self.resources = resources or []
        self.predecessors = predecessors or []
        self.description = description

        # Schedule fields
        self.slack: int = 0
        self.asap_start: Optional[int] = None
        self.asap_finish: Optional[int] = None
        self.alap_start: Optional[int] = None
        self.alap_finish: Optional[int] = None
        self.scheduled_start: Optional[int] = None
        self.scheduled_finish: Optional[int] = None
        self.on_critical_chain: bool = False
        self.successors: List[str] = []

        # CCPM and execution fields
        self.original_duration: Optional[int] = None
        self.ccpm_duration: Optional[int] = None
        self.safety_removed: int = 0
        self.execution_status: TaskStatus = TaskStatus.NOT_STARTED
        self.actual_start: Optional[int] = None
        self.actual_finish: Optional[int] = None

    def apply_ccpm_safety_reduction(self, safety_factor: float = 0.5) -> None:
        """Reduces task duration for CCPM planning and stores original."""
        if self.original_duration is None:
            self.original_duration = self.duration

        # Use floor for aggressive scheduling
        self.ccpm_duration = int(self.original_duration * (1 - safety_factor))
        self.safety_removed = self.original_duration - self.ccpm_duration
        self.duration = self.ccpm_duration

    def revert_to_original_duration(self) -> bool:
        """Reverts task duration to its original value for replanning."""
        if self.original_duration is not None:
            self.duration = self.original_duration
            self.original_duration = None
            self.ccpm_duration = None
            self.safety_removed = 0
            return True
        return False

    def update_progress(
        self,
        status: TaskStatus,
        actual_start: Optional[int] = None,
        actual_finish: Optional[int] = None
    ) -> None:
        """Updates the task's execution status and actual times."""
        self.execution_status = status
        if actual_start is not None:
            self.actual_start = actual_start
        if actual_finish is not None:
            self.actual_finish = actual_finish

    def calculate_delay(self) -> int:
        """Calculates delay based on scheduled vs. actual finish."""
        if self.actual_finish is not None and self.scheduled_finish is not None:
            delay = self.actual_finish - self.scheduled_finish
            # Consider only delays, not early finishes
            return max(0, delay)
        return 0

    def get_predecessors(self) -> List[str]:
        return self.predecessors

    def get_successors(self) -> List[str]:
        return self.successors

    def to_dict(self) -> Dict[str, Any]:
        """Serializes task to a dictionary."""
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
            "original_duration": self.original_duration,
            "ccpm_duration": self.ccpm_duration,
            "safety_removed": self.safety_removed,
            "execution_status": self.execution_status.name,
            "actual_start": self.actual_start,
            "actual_finish": self.actual_finish,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserializes task from a dictionary."""
        resources = [r.strip() for r in str(data.get("resources", "")).split(",") if r.strip()]
        preds = [p.strip() for p in str(data.get("predecessors", "")).split(",") if p.strip()]

        t = cls(
            task_id=data["id"],
            name=data.get("name", data["id"]),
            duration=int(data["duration"]),
            resources=resources,
            predecessors=preds,
            description=data.get("description", ""),
        )

        # Restore all fields
        t.asap_start = data.get("asap_start")
        t.asap_finish = data.get("asap_finish")
        t.alap_start = data.get("alap_start")
        t.alap_finish = data.get("alap_finish")
        t.scheduled_start = data.get("scheduled_start")
        t.scheduled_finish = data.get("scheduled_finish")
        t.on_critical_chain = bool(data.get("on_critical_chain", False))
        t.original_duration = data.get("original_duration")
        t.ccpm_duration = data.get("ccpm_duration")
        t.safety_removed = data.get("safety_removed", 0)

        status_str = data.get("execution_status")
        if status_str and status_str in TaskStatus.__members__:
            t.execution_status = TaskStatus[status_str]

        t.actual_start = data.get("actual_start")
        t.actual_finish = data.get("actual_finish")

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
        self.buffer_type = "generic"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["linked_task"] = self.linked_task
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Buffer":
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
        self.buffer_type = "project"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["delivery_date"] = self.delivery_date
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectBuffer":
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
    def __init__(
        self,
        buffer_id: str,
        name: str,
        duration: int,
        linked_task: Optional[str] = None,
        description: str = "",
    ):
        super().__init__(buffer_id, name, duration, linked_task, description)
        self.buffer_type = "feeding"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedingBuffer":
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


class SafetyTracker:
    """Track safety time removed from tasks for buffer calculations"""
    def __init__(self):
        self.removed_safety: Dict[str, int] = {}
        self.chain_safety: Dict[str, int] = {}

    def calculate_chain_safety(self, chain_tasks: List[str]) -> int:
        """Sum removed safety for a chain of tasks"""
        return sum(self.removed_safety.get(tid, 0) for tid in chain_tasks)


def detect_feeding_chains(
    tasks: Dict[str, Task], critical_chain: List[str]
) -> Dict[str, List[str]]:
    """
    For each critical chain task, identify feeding chains that merge into it.
    Returns mapping: {critical_task_id: [feeding_chain_task_ids]}
    """
    critical_chain_set = set(critical_chain)
    feeding_chains: Dict[str, List[str]] = {}

    for task_id, task in tasks.items():
        if task_id in critical_chain_set:
            continue

        # Find the first critical chain task this task feeds into
        q = list(task.successors)
        visited = set(q)
        merge_point = None

        while q:
            curr_id = q.pop(0)
            if curr_id in critical_chain_set:
                merge_point = curr_id
                break

            for succ_id in tasks[curr_id].successors:
                if succ_id not in visited:
                    q.append(succ_id)
                    visited.add(succ_id)

        if merge_point:
            # We found that `task_id` is part of a chain that feeds into `merge_point`.
            # Now, trace back from `task_id` to find the whole chain.
            chain = []
            q_back = [task_id]
            visited_back = {task_id}
            while q_back:
                curr_back_id = q_back.pop(0)
                if tasks[curr_back_id].on_critical_chain:
                    continue
                chain.append(curr_back_id)
                for pred_id in tasks[curr_back_id].predecessors:
                    if pred_id not in visited_back:
                        q_back.append(pred_id)
                        visited_back.add(pred_id)

            if merge_point not in feeding_chains:
                feeding_chains[merge_point] = []
            feeding_chains[merge_point].extend(chain)

    # Remove duplicates
    for merge_point in feeding_chains:
        feeding_chains[merge_point] = list(dict.fromkeys(feeding_chains[merge_point]))

    return feeding_chains


def integrate_buffers_into_schedule(
    tasks: Dict[str, Task],
    project_buffer: Optional[ProjectBuffer],
    feeding_buffers: Dict[str, FeedingBuffer],
) -> None:
    """
    Insert buffer tasks into project network with proper dependencies.
    This function modifies the tasks dictionary in-place.
    """
    # Add feeding buffers to the main task list
    for buffer in feeding_buffers.values():
        tasks[buffer.id] = buffer

    # Wire feeding buffers into the graph
    for buffer in feeding_buffers.values():
        if not buffer.linked_task:
            continue
        merge_point_task = tasks[buffer.linked_task]

        # The buffer's successor is the merge point task
        buffer.successors = [buffer.linked_task]

        # Find the tasks at the end of the feeding chain
        # These are the tasks that originally fed into the merge point
        feeding_chain_endpoints = [
            tid for tid, task in tasks.items()
            if buffer.linked_task in task.successors
            and not task.on_critical_chain
            and not isinstance(task, Buffer)
        ]

        buffer.predecessors = feeding_chain_endpoints

        # Re-wire the graph
        for pred_id in feeding_chain_endpoints:
            pred_task = tasks[pred_id]
            # The predecessor's successor is now the buffer, not the merge point
            pred_task.successors = [s for s in pred_task.successors if s != buffer.linked_task]
            pred_task.successors.append(buffer.id)

        # The merge point's predecessor is now the buffer
        merge_point_task.predecessors = [p for p in merge_point_task.predecessors if p not in feeding_chain_endpoints]
        merge_point_task.predecessors.append(buffer.id)

    # Wire in the project buffer
    if project_buffer:
        # Find the last task(s) in the project (tasks with no successors)
        # and make them predecessors to the project buffer.
        last_task_ids = [tid for tid, task in tasks.items() if not task.successors and task.id != project_buffer.id]

        project_buffer.predecessors = last_task_ids
        for tid in last_task_ids:
            tasks[tid].successors.append(project_buffer.id)

        tasks[project_buffer.id] = project_buffer


def calculate_feeding_buffers(
    feeding_chains: Dict[str, List[str]],
    safety_tracker: SafetyTracker,
    buffer_factor: float = 0.5,
) -> Dict[str, FeedingBuffer]:
    """Calculate and create feeding buffer objects"""
    buffers: Dict[str, FeedingBuffer] = {}
    for merge_point, chain in feeding_chains.items():
        buffer_size = round(safety_tracker.calculate_chain_safety(chain) * buffer_factor)
        if buffer_size > 0:
            buffer_id = f"FeedingBuffer-{merge_point}"
            buffer_name = f"Feeding Buffer for {merge_point}"
            buffer = FeedingBuffer(buffer_id, buffer_name, buffer_size, linked_task=merge_point)
            buffers[buffer_id] = buffer
    return buffers


def calculate_project_buffer(
    critical_chain_tasks: List[str],
    safety_tracker: SafetyTracker,
    buffer_factor: float = 0.5,
) -> int:
    """
    Calculate project buffer size (typically 50% of removed critical chain safety).
    """
    removed_safety = safety_tracker.calculate_chain_safety(critical_chain_tasks)
    return round(removed_safety * buffer_factor)


def insert_project_buffer(
    tasks: Dict[str, Task],
    critical_chain: List[str],
    buffer_size: int,
) -> ProjectBuffer:
    """Creates a project buffer task object."""
    buffer_id = "ProjectBuffer"
    buffer_name = "Project Buffer"
    buffer = ProjectBuffer(buffer_id, buffer_name, buffer_size)
    return buffer


def calculate_aggressive_durations(
    tasks: Dict[str, Task],
    safety_tracker: SafetyTracker,
    safety_factor: float = 0.5
) -> None:
    """
    Applies CCPM safety reduction to all tasks and updates the safety tracker.
    """
    for task in tasks.values():
        # This method now lives on the Task object
        task.apply_ccpm_safety_reduction(safety_factor)
        safety_tracker.removed_safety[task.id] = task.safety_removed


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


def detect_cycles(tasks: List[Task]) -> None:
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


def diagnose_unschedulable(
    tasks: Dict[str, Task],
    resources: Dict[str, Resource],
    project_calendar: ProjectCalendar,
    forward_map: Dict[str, Tuple[int, int]],
) -> None:
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
                if task.alap_finish is not None and pred_end > task.alap_finish:
                    reasons.append(
                        f"predecessor '{pid}' finishes at day {pred_end}, "
                        f"after task's latest finish {task.alap_finish}"
                    )

        # Resource availability blocking
        if task.alap_start is not None and task.alap_finish is not None:
            for res_id in task.resources:
                if res_id not in resources:
                    reasons.append(f"resource '{res_id}' not found")
                    continue
                resource = resources[res_id]
                for day in range(task.alap_start, task.alap_finish + 1):
                    if not resource.is_available(day, project_calendar):
                        reasons.append(f"resource '{resource.name}' unavailable on day {day}")
                        break  # No need to check other days for this resource

        if reasons:
            print(f"Task '{tid}' is unschedulable because:")
            for r in reasons:
                print(f"  - {r}")


def find_cycles(tasks: List[Task]) -> List[List[str]]:
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


def identify_critical_chain(
    tasks: Dict[str, Task],
    schedule: Dict[str, Tuple[int, int]],
) -> List[str]:
    """
    Identifies the critical chain considering both task and resource dependencies.
    The resource dependencies are inferred from the provided schedule.
    """
    # 1. Build a combined dependency graph (predecessors + resources)
    adj: Dict[str, List[str]] = {tid: [] for tid in tasks}
    in_degree: Dict[str, int] = {tid: 0 for tid in tasks}

    # Add predecessor dependencies
    for tid, task in tasks.items():
        for pred_id in task.predecessors:
            adj[pred_id].append(tid)
            in_degree[tid] += 1

    # Add resource dependencies from the schedule
    resource_map = build_resource_dependency_graph(tasks)
    for res_id, task_ids in resource_map.items():
        if len(task_ids) <= 1:
            continue
        # Sort tasks using this resource by their scheduled start time
        sorted_tasks = sorted(task_ids, key=lambda tid: schedule[tid][0])
        for i in range(len(sorted_tasks) - 1):
            u, v = sorted_tasks[i], sorted_tasks[i+1]
            # Add a resource dependency edge
            adj[u].append(v)
            in_degree[v] += 1

    # 2. Find the longest path in the combined graph (using task durations as weights)
    q = [tid for tid, deg in in_degree.items() if deg == 0]
    dist: Dict[str, int] = {tid: 0 for tid in tasks}
    path_pred: Dict[str, Optional[str]] = {tid: None for tid in tasks}

    for tid in q:
        dist[tid] = tasks[tid].duration

    while q:
        u = q.pop(0)
        for v in adj[u]:
            new_dist = dist[u] + tasks[v].duration
            if new_dist > dist[v]:
                dist[v] = new_dist
                path_pred[v] = u

            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    # 3. Find the end of the longest path and backtrack to reconstruct it
    end_task_id = max(dist, key=lambda k: dist[k])
    critical_chain = []
    curr = end_task_id
    while curr:
        critical_chain.append(curr)
        tasks[curr].on_critical_chain = True
        curr = path_pred[curr]

    critical_chain.reverse()
    return critical_chain


def build_resource_dependency_graph(tasks: Dict[str, Task]) -> Dict[str, List[str]]:
    """
    Groups tasks by the resources they require.
    Returns a dictionary where keys are resource IDs and values are lists of task IDs.
    """
    resource_map: Dict[str, List[str]] = {}
    for task_id, task in tasks.items():
        for resource_id in task.resources:
            if resource_id not in resource_map:
                resource_map[resource_id] = []
            resource_map[resource_id].append(task_id)
    return resource_map


def detect_circular_dependencies(tasks: Dict[str, "Task"]) -> List[List[str]]:
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

    asap_schedule: Dict[str, Tuple[int, int]] = {}
    for tid, t in tasks.items():
        if t.asap_start is None or t.asap_finish is None:
            # This should not be reachable due to the logic above
            raise RuntimeError(f"Task {tid} was not scheduled in ASAP pass.")
        asap_schedule[tid] = (t.asap_start, t.asap_finish)
    return asap_schedule


# ------------------------
# ALAP backward pass (ignores resources)
# ------------------------
def compute_alap(
    tasks: Dict[str, Task], project_calendar: ProjectCalendar, delivery_day: int
) -> None:  # type: ignore
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
            succ_starts = [tasks[s].alap_start for s in succs[tid] if tasks[s].alap_start is not None]
            if not succ_starts:
                # This case should ideally not be reached in a valid graph
                finish = delivery_day
            else:
                finish = min(s - 1 for s in succ_starts)  # type: ignore
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
    task_lookup = tasks

    # --- Detect circular dependencies first ---
    visited, rec_stack = set(), set()

    def dfs(task_id: str) -> bool:
        visited.add(task_id)
        rec_stack.add(task_id)
        for pred_id in task_lookup[task_id].predecessors:
            if pred_id not in task_lookup:
                continue
            if pred_id not in visited and dfs(pred_id):
                return True
            elif pred_id in rec_stack:
                raise RuntimeError(f"Circular dependency detected involving {pred_id}")
        rec_stack.remove(task_id)
        return False

    for tid in task_lookup:
        if tid not in visited:
            dfs(tid)

    # --- Initialize scheduling map ---
    schedule: Dict[str, Tuple[Optional[int], Optional[int]]] = {tid: (None, None) for tid in task_lookup}

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
                succ_starts = [schedule[sid][0] for sid in task.successors]
                if not succ_starts or any(s is None for s in succ_starts):
                    if diagnostic:
                        print(f"Task {tid} blocked: successors have no start times yet.")
                    continue
                min_succ_start = min(s for s in succ_starts if s is not None)
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

    # At this point, all tasks should be scheduled and have non-None start/finish times.
    # We can safely cast the result to the expected return type.
    final_schedule: Dict[str, Tuple[int, int]] = {}
    for tid, times in schedule.items():
        s, f = times
        if s is None or f is None:
            # This should not happen if `unscheduled` is empty.
            raise RuntimeError(f"Task {tid} was not scheduled but error was not raised.")
        final_schedule[tid] = (s, f)

    return final_schedule


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
    order = topological_sort(task_lookup)

    for tid in order:
        task = task_lookup[tid]

        # Earliest start = max of predecessors' finish days + 1
        earliest_start = 0
        if task.predecessors:
            # Predecessors are guaranteed to be in asap_schedule due to topological sort
            earliest_start = max(asap_schedule[p][1] for p in task.predecessors) + 1

        # Schedule ASAP (no resource check for this pass)
        start_day = project_calendar.next_working_day(earliest_start)
        finish_day = project_calendar.add_working_days(start_day, task.duration) - 1

        asap_schedule[tid] = (start_day, finish_day)

        if diagnostic:
            print(f"ASAP: Scheduled Task {tid} {start_day}-{finish_day}")

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
# CCPM Pipeline
# ------------------------
@dataclass
class CCPMScheduleResult:
    """Complete CCPM scheduling results"""
    tasks: Dict[str, Task]
    critical_chain: List[str]
    project_buffer: Optional[ProjectBuffer]
    feeding_buffers: Dict[str, FeedingBuffer]
    safety_tracker: SafetyTracker
    schedule_statistics: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Export complete schedule to DataFrame"""
        return tasks_to_df(list(self.tasks.values()))


def schedule_with_ccpm(
    tasks: Union[List[Task], Dict[str, Task]],
    resources: Dict[str, Resource],
    project_calendar: ProjectCalendar,
    max_day: int,
    safety_factor: float = 0.5,
    buffer_factor: float = 0.5,
) -> CCPMScheduleResult:
    """
    Complete CCPM scheduling pipeline.
    """
    # Normalize tasks to dict
    if isinstance(tasks, list):
        task_lookup = {t.id: t for t in tasks}
    else:
        task_lookup = tasks

    # 1. Initial ASAP/ALAP scheduling
    initial_schedule = schedule_tasks(task_lookup, resources, project_calendar, max_day)

    # 2. Critical chain identification
    critical_chain = identify_critical_chain(task_lookup, initial_schedule["ALAP"])

    # 3. Duration adjustment (safety removal)
    safety_tracker = SafetyTracker()
    calculate_aggressive_durations(task_lookup, safety_tracker, safety_factor)

    # 4. Buffer calculation
    project_buffer_size = calculate_project_buffer(critical_chain, safety_tracker, buffer_factor)
    project_buffer = insert_project_buffer(task_lookup, critical_chain, project_buffer_size)

    feeding_chains = detect_feeding_chains(task_lookup, critical_chain)
    feeding_buffers = calculate_feeding_buffers(feeding_chains, safety_tracker, buffer_factor)

    # 5. Buffer integration
    integrate_buffers_into_schedule(task_lookup, project_buffer, feeding_buffers)

    # 6. Final scheduling with buffers
    final_schedule = schedule_tasks(task_lookup, resources, project_calendar, max_day)

    # 7. Validation and diagnostics
    stats = analyze_schedule_quality(
        task_lookup, final_schedule["ASAP"], final_schedule["ALAP"], resources
    )

    return CCPMScheduleResult(
        tasks=task_lookup,
        critical_chain=critical_chain,
        project_buffer=project_buffer,
        feeding_buffers=feeding_buffers,
        safety_tracker=safety_tracker,
        schedule_statistics=stats,
    )


# ------------------------
# DataFrame helpers (your existing helpers kept)
# ------------------------
def analyze_schedule_quality(
    tasks: Dict[str, Task],
    asap_schedule: Dict[str, Tuple[int, int]],
    alap_schedule: Dict[str, Tuple[int, int]],
    resources: Dict[str, Resource],
) -> Dict[str, Any]:
    """
    Provides comprehensive schedule analysis.
    """
    # Slack Analysis
    slacks = {
        tid: alap_schedule[tid][0] - asap_schedule[tid][0] for tid in tasks
    }
    avg_slack = sum(slacks.values()) / len(slacks) if slacks else 0

    # Resource Utilization
    resource_allocations: Dict[str, Dict[int, int]] = {res_id: {} for res_id in resources}
    max_day = 0
    for tid, (start, end) in alap_schedule.items():
        task = tasks[tid]
        if end > max_day:
            max_day = end
        for res_id in task.resources:
            for day in range(start, end + 1):
                allocations = resource_allocations.setdefault(res_id, {})
                allocations[day] = allocations.get(day, 0) + 1 # Assuming 1 unit per task

    total_capacity = 0
    total_allocated = 0
    for res_id, res in resources.items():
        total_capacity += res.capacity_per_day * (max_day + 1)
        total_allocated += sum(resource_allocations.get(res_id, {}).values())

    avg_utilization = (total_allocated / total_capacity) * 100 if total_capacity > 0 else 0

    return {
        "average_slack": avg_slack,
        "slack_distribution": slacks,
        "average_resource_utilization_percent": avg_utilization,
        "project_finish_day": max_day,
    }


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
            task_id=str(row["id"]),
            name=str(row.get("name", row["id"])),
            duration=int(row["duration"]),
            resources=resources,
            predecessors=preds,
            description=str(row.get("description", "")),
        )
        def to_optional_int(val: Any) -> Optional[int]:
            if pd.isna(val):
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        t.asap_start = to_optional_int(row.get("asap_start"))
        t.asap_finish = to_optional_int(row.get("asap_finish"))
        t.alap_start = to_optional_int(row.get("alap_start"))
        t.alap_finish = to_optional_int(row.get("alap_finish"))
        t.scheduled_start = to_optional_int(row.get("scheduled_start"))
        t.scheduled_finish = to_optional_int(row.get("scheduled_finish"))
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
