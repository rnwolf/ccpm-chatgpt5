import pandas as pd
from ccpm_module import Task, Resource, ProjectCalendar

def create_scenario_calendar_impact():
    """
    Scenario 1: A simple linear project to demonstrate the impact of
    weekends and holidays on the schedule.
    """
    tasks = [
        {'id': 'A', 'name': 'Task A', 'duration': 5, 'predecessors': '', 'resources': 'R1'},
        {'id': 'B', 'name': 'Task B', 'duration': 5, 'predecessors': 'A', 'resources': 'R1'},
        {'id': 'C', 'name': 'Task C', 'duration': 5, 'predecessors': 'B', 'resources': 'R1'},
    ]
    tasks_df = pd.DataFrame(tasks)

    resources = [
        {'id': 'R1', 'name': 'Worker 1', 'capacity_per_day': 1, 'non_working_days': ''},
    ]
    resources_df = pd.DataFrame(resources)

    # Weekends (Sat/Sun) and a holiday on day 8
    non_working_days = [d for d in range(0, 30) if d % 7 in (5, 6)] + [8]
    calendar_df = pd.DataFrame(non_working_days, columns=['non_working_days'])

    with pd.ExcelWriter('scenario_calendar_impact.xlsx') as writer:
        tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
        resources_df.to_excel(writer, sheet_name='Resources', index=False)
        calendar_df.to_excel(writer, sheet_name='ProjectCalendar', index=False)

def create_scenario_resource_vacation():
    """
    Scenario 2: A project with resource contention where a key resource
    has specific vacation days.
    """
    tasks = [
        {'id': 'A', 'name': 'Setup', 'duration': 3, 'predecessors': '', 'resources': 'R1'},
        {'id': 'B', 'name': 'Parallel Task 1', 'duration': 10, 'predecessors': 'A', 'resources': 'R2'},
        {'id': 'C', 'name': 'Parallel Task 2', 'duration': 8, 'predecessors': 'A', 'resources': 'R2'},
        {'id': 'D', 'name': 'Merge', 'duration': 3, 'predecessors': 'B,C', 'resources': 'R1'},
    ]
    tasks_df = pd.DataFrame(tasks)

    resources = [
        {'id': 'R1', 'name': 'Lead', 'capacity_per_day': 1, 'non_working_days': ''},
        # R2 has vacation from day 10 to 14
        {'id': 'R2', 'name': 'Specialist', 'capacity_per_day': 1, 'non_working_days': '10,11,12,13,14'},
    ]
    resources_df = pd.DataFrame(resources)

    # Weekends (Sat/Sun)
    non_working_days = [d for d in range(0, 40) if d % 7 in (5, 6)]
    calendar_df = pd.DataFrame(non_working_days, columns=['non_working_days'])

    with pd.ExcelWriter('scenario_resource_vacation.xlsx') as writer:
        tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
        resources_df.to_excel(writer, sheet_name='Resources', index=False)
        calendar_df.to_excel(writer, sheet_name='ProjectCalendar', index=False)

def create_scenario_multi_calendar():
    """
    Scenario 3: A complex project with feeding chains where different
    resources have their own unique non-working days.
    """
    tasks = [
        {'id': 'CC1', 'name': 'Critical 1', 'duration': 10, 'predecessors': '', 'resources': 'R1'},
        {'id': 'CC2', 'name': 'Critical 2', 'duration': 15, 'predecessors': 'CC1,FC1B', 'resources': 'R1'},
        {'id': 'CC3', 'name': 'Critical 3', 'duration': 10, 'predecessors': 'CC2,FC2B', 'resources': 'R1'},
        {'id': 'FC1A', 'name': 'Feed 1A', 'duration': 8, 'predecessors': '', 'resources': 'R2'},
        {'id': 'FC1B', 'name': 'Feed 1B', 'duration': 7, 'predecessors': 'FC1A', 'resources': 'R2'},
        {'id': 'FC2A', 'name': 'Feed 2A', 'duration': 6, 'predecessors': '', 'resources': 'R3'},
        {'id': 'FC2B', 'name': 'Feed 2B', 'duration': 9, 'predecessors': 'FC2A', 'resources': 'R3'},
    ]
    tasks_df = pd.DataFrame(tasks)

    resources = [
        {'id': 'R1', 'name': 'Core Team', 'capacity_per_day': 1, 'non_working_days': ''},
        {'id': 'R2', 'name': 'Team A', 'capacity_per_day': 1, 'non_working_days': '20,21'}, # Team A off
        {'id': 'R3', 'name': 'Team B', 'capacity_per_day': 1, 'non_working_days': '30,31'}, # Team B off
    ]
    resources_df = pd.DataFrame(resources)

    # Weekends (Sat/Sun)
    non_working_days = [d for d in range(0, 60) if d % 7 in (5, 6)]
    calendar_df = pd.DataFrame(non_working_days, columns=['non_working_days'])

    with pd.ExcelWriter('scenario_multi_calendar.xlsx') as writer:
        tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
        resources_df.to_excel(writer, sheet_name='Resources', index=False)
        calendar_df.to_excel(writer, sheet_name='ProjectCalendar', index=False)

if __name__ == "__main__":
    create_scenario_calendar_impact()
    create_scenario_resource_vacation()
    create_scenario_multi_calendar()
    print("Successfully created all three Excel scenario files.")
