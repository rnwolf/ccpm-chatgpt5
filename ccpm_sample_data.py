# ccpm_sample_data.py

from ccpm_module import Task, Resource, ProjectCalendar

def get_sample_tasks():
    """Returns a list of sample tasks for template generation."""
    tasks = [
        Task(
            task_id="T1",
            name="Project Initiation",
            duration=5,
            description="Define project scope, goals, and stakeholders."
        ),
        Task(
            task_id="T2",
            name="Requirement Gathering",
            duration=10,
            predecessors=["T1"],
            resources=["BA1"],
            description="Gather and document project requirements from stakeholders."
        ),
        Task(
            task_id="T3",
            name="Design",
            duration=8,
            predecessors=["T2"],
            resources=["DE1"],
            description="Create system design and architecture."
        ),
        Task(
            task_id="T4",
            name="Development - Module A",
            duration=15,
            predecessors=["T3"],
            resources=["DV1", "DV2"],
            description="Develop and unit test Module A."
        ),
        Task(
            task_id="T5",
            name="Development - Module B",
            duration=12,
            predecessors=["T3"],
            resources=["DV2"],
            description="Develop and unit test Module B."
        ),
        Task(
            task_id="T6",
            name="Integration",
            duration=5,
            predecessors=["T4", "T5"],
            resources=["DV1", "DV2"],
            description="Integrate Module A and B."
        ),
        Task(
            task_id="T7",
            name="Testing",
            duration=10,
            predecessors=["T6"],
            resources=["QA1"],
            description="Perform system and user acceptance testing."
        ),
        Task(
            task_id="T8",
            name="Deployment",
            duration=3,
            predecessors=["T7"],
            description="Deploy the application to production."
        ),
    ]
    return tasks

def get_sample_resources():
    """Returns a list of sample resources for template generation."""
    resources = [
        Resource(resource_id="BA1", name="Business Analyst"),
        Resource(resource_id="DE1", name="Designer"),
        Resource(resource_id="DV1", name="Developer 1"),
        Resource(resource_id="DV2", name="Developer 2"),
        Resource(resource_id="QA1", name="QA Engineer"),
    ]
    return resources

def get_sample_calendar():
    """Returns a sample project calendar for template generation."""
    # Example: Weekends are non-working days
    non_working_days = [d for d in range(0, 100) if d % 7 in (5, 6)]
    calendar = ProjectCalendar(non_working_days=non_working_days)
    return calendar
