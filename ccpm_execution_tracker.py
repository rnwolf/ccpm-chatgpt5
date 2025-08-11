from dataclasses import dataclass
from typing import Optional, Dict, Any

from ccpm_module import CCPMScheduleResult, TaskStatus


@dataclass
class ProgressUpdate:
    """Represents a single progress update for a task."""
    task_id: str
    status: TaskStatus
    actual_start: Optional[int] = None
    actual_finish: Optional[int] = None
    percent_complete: Optional[float] = None


class ProjectExecutionTracker:
    """
    Tracks the execution of a project, applying progress updates and
    calculating buffer consumption.
    """

    def __init__(self, schedule_result: CCPMScheduleResult):
        self.schedule_result = schedule_result
        self.tasks = schedule_result.tasks

    def update_task_progress(self, update: ProgressUpdate) -> None:
        """Applies a progress update to a task in the project."""
        if update.task_id not in self.tasks:
            # Or handle this with a warning/logging
            raise ValueError(f"Task with ID '{update.task_id}' not found in project.")

        task = self.tasks[update.task_id]

        # Use the task's own update method
        task.update_progress(
            status=update.status,
            actual_start=update.actual_start,
            actual_finish=update.actual_finish,
        )

        # If percent complete is provided, we might want to update the
        # expected finish date, but that's a more advanced feature
        # for replanning, which is out of scope for now.
        if update.percent_complete is not None:
            # For now, we just note it. In a future implementation, this
            # could be used to predict delays.
            pass

    def get_buffer_statuses(self, current_date: int) -> Dict[str, Dict[str, Any]]:
        """
        Calculates and returns the status of all buffers at a given date.
        """
        from fever_chart_system import BufferConsumptionCalculator

        calculator = BufferConsumptionCalculator(self.schedule_result)
        statuses = {}

        # Project Buffer
        if self.schedule_result.project_buffer:
            pb = self.schedule_result.project_buffer
            planned = calculator.calculate_planned_consumption(pb.id, current_date)
            actual = calculator.calculate_actual_consumption(pb.id, current_date, self.tasks)
            statuses[pb.name] = {
                "type": "Project",
                "size": pb.duration,
                "planned_consumption": planned,
                "actual_consumption": actual,
            }

        # Feeding Buffers
        for fb in self.schedule_result.feeding_buffers.values():
            planned = calculator.calculate_planned_consumption(fb.id, current_date)
            actual = calculator.calculate_actual_consumption(fb.id, current_date, self.tasks)
            statuses[fb.name] = {
                "type": "Feeding",
                "size": fb.duration,
                "planned_consumption": planned,
                "actual_consumption": actual,
            }

        return statuses
