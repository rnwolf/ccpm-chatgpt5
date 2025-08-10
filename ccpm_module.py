# ccpm_module.py

from typing import List, Dict, Optional
from datetime import date, timedelta
import pandas as pd


class ProjectCalendar:
    """
    Project-wide calendar for defining global non-working days.
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
    Represents a named resource and its availability.
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
        self.resources = resources or []
        self.predecessors = predecessors or []
        self.description = description
        self.start_day: Optional[int] = None
        self.end_day: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "resources": self.resources,
            "predecessors": self.predecessors,
            "description": self.description,
            "start_day": self.start_day,
            "end_day": self.end_day,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        task = cls(
            task_id=data["id"],
            name=data["name"],
            duration=data["duration"],
            resources=data.get("resources", []),
            predecessors=data.get("predecessors", []),
            description=data.get("description", ""),
        )
        task.start_day = data.get("start_day")
        task.end_day = data.get("end_day")
        return task


class Buffer(Task):
    """
    Base class for buffers in CCPM.
    """

    def __init__(
        self,
        buffer_id: str,
        name: str,
        duration: int,
        description: str = "",
        linked_task: Optional[str] = None,
    ):
        super().__init__(buffer_id, name, duration, [], [], description)
        self.linked_task = linked_task

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["linked_task"] = self.linked_task
        return data

    @classmethod
    def from_dict(cls, data: Dict):
        buf = cls(
            buffer_id=data["id"],
            name=data["name"],
            duration=data["duration"],
            description=data.get("description", ""),
            linked_task=data.get("linked_task"),
        )
        buf.start_day = data.get("start_day")
        buf.end_day = data.get("end_day")
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
        super().__init__(buffer_id, name, duration, description)
        self.delivery_date = delivery_date

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["delivery_date"] = self.delivery_date
        return data

    @classmethod
    def from_dict(cls, data: Dict):
        buf = cls(
            buffer_id=data["id"],
            name=data["name"],
            duration=data["duration"],
            delivery_date=data.get("delivery_date"),
            description=data.get("description", ""),
        )
        buf.start_day = data.get("start_day")
        buf.end_day = data.get("end_day")
        return buf


class FeedingBuffer(Buffer):
    pass


# -----------------
# Simple scheduler
# -----------------
def schedule_tasks(
    tasks: List[Task], resources: List[Resource], calendar: ProjectCalendar
):
    """
    A very basic forward-pass scheduler that assigns earliest start dates
    without proper CCPM resource leveling.
    """
    task_map = {t.id: t for t in tasks}
    scheduled = set()

    while len(scheduled) < len(tasks):
        for task in tasks:
            if task.id in scheduled:
                continue
            if all(pred in scheduled for pred in task.predecessors):
                earliest_start = 0
                for pred in task.predecessors:
                    pred_task = task_map[pred]
                    earliest_start = max(earliest_start, pred_task.end_day or 0)

                start_day = earliest_start
                day_count = 0
                while day_count < task.duration:
                    if all(
                        res.is_available(start_day, calendar)
                        for res in (r for r in resources if r.id in task.resources)
                    ):
                        day_count += 1
                    start_day += 1

                task.start_day = earliest_start
                task.end_day = start_day
                scheduled.add(task.id)

    return tasks


def resources_to_availability_df(resources: list, max_day: int = 31):
    """
    Create a DataFrame with each resource as a column (using resource ID as column name)
    and each row as a zero-based day index. 1 = non-working, 0 = working.
    """
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
    df.attrs["id_to_name"] = id_to_name  # Store mapping for reference if needed
    return df


def resources_df_to_availability(df: pd.DataFrame) -> list:
    """
    Convert a DataFrame (days as rows, resource IDs as columns, 1=non-working, 0=working)
    back to a list of Resource objects with updated non_working_days.
    Uses the id_to_name mapping in df.attrs if present to restore resource names.
    """
    resources = []
    id_to_name = df.attrs.get("id_to_name", {})
    for col in df.columns:
        non_working_days = df.index[df[col] == 1].tolist()
        name = id_to_name.get(col, col)
        resources.append(
            Resource(resource_id=col, name=name, non_working_days=non_working_days)
        )
    return resources


def tasks_to_df(tasks: list) -> pd.DataFrame:
    """
    Convert a list of Task objects to a DataFrame for easy editing.
    List fields (resources, predecessors) are joined as comma-separated strings.
    """
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
                "start_day": t.start_day,
                "end_day": t.end_day,
            }
        )
    return pd.DataFrame(rows)


def df_to_tasks(df: pd.DataFrame) -> list:
    """
    Convert a DataFrame (with columns matching tasks_to_df) back to a list of Task objects.
    Assumes resources and predecessors are comma-separated strings.
    """
    tasks = []
    for _, row in df.iterrows():
        resources = [
            r.strip() for r in str(row.get("resources", "")).split(",") if r.strip()
        ]
        predecessors = [
            p.strip() for p in str(row.get("predecessors", "")).split(",") if p.strip()
        ]
        task = Task(
            task_id=row["id"],
            name=row["name"],
            duration=int(row["duration"]),
            resources=resources,
            predecessors=predecessors,
            description=row.get("description", ""),
        )
        task.start_day = row.get("start_day")
        task.end_day = row.get("end_day")
        tasks.append(task)
    return tasks


# -----------------
# Example
# -----------------
if __name__ == "__main__":
    calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])  # weekends

    resources_dict_list = [
        {"id": "R1", "name": "Alice", "non_working_days": [6, 7]},
        {"id": "R2", "name": "Bob", "non_working_days": []},
        {"id": "R3", "name": "Charlie", "non_working_days": [7]},
    ]
    df = pd.DataFrame(resources_dict_list)
    print(df)

    # Update non_working_days for the resource with id "R1" to simulate a human edit
    df.loc[df["id"] == "R1", "non_working_days"] = [[2, 6, 7]]

    # Convert DataFrame to Resource objects
    resources = resources_df_to_availability(df)

    # Alternatively, create Resource objects directly
    # resources = [
    #     Resource("R1", "Alice", non_working_days=[7]),
    #     Resource("R2", "Bob"),
    #     Resource("R3", "Charlie", non_working_days=[7]),
    # ]

    # Convert resources to DataFrame for easy viewing/editing
    df = resources_to_availability_df(resources, max_day=14)
    print(df)

    # Example tasks

    tasks_dict_list = [
        {
            "id": "T1",
            "name": "Design",
            "duration": 4,
            "resources": ["R1"],
            "predecessors": [],
            "description": "",
            "start_day": 0,
            "end_day": 4,
        },
        {
            "id": "T2",
            "name": "Develop",
            "duration": 5,
            "resources": ["R1", "R2"],
            "predecessors": ["T1"],
            "description": "",
            "start_day": 4,
            "end_day": 12,
        },
        {
            "id": "T3",
            "name": "Test",
            "duration": 2,
            "resources": ["R2"],
            "predecessors": ["T2"],
            "description": "",
            "start_day": 12,
            "end_day": 16,
        },
    ]

    tasks_df = pd.DataFrame(tasks_dict_list)
    print(tasks_df)

    # Simulate human edit: change duration of "Design" to 6
    tasks_df.loc[tasks_df["id"] == "T1", "duration"] = 6

    tasks = df_to_tasks(tasks_df)

    # Alternatively, create Task objects directly
    # tasks = [
    #     Task("T1", "Design", 3, ["R1"]),
    #     Task("T2", "Develop", 5, ["R1", "R2"], predecessors=["T1"]),
    #     Task("T3", "Test", 2, ["R2"], predecessors=["T2"]),
    # ]

    scheduled_tasks = schedule_tasks(tasks, resources, calendar)

    for t in scheduled_tasks:
        print(t.to_dict())

    # Convert scheduled tasks to DataFrame for easy viewing/editing
    plan_df = tasks_to_df(scheduled_tasks)
    print(plan_df)
    plan_df.to_csv("scheduled_tasks.csv", index=False)
    print("Scheduled tasks saved to 'scheduled_tasks.csv'")
