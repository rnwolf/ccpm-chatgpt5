#!/usr/bin/env python3
"""
CCPM Command Line Interface
Production-ready tool for CCPM project management
"""

import click
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

# Import our CCPM modules
from ccpm_module import (
    Task, Resource, ProjectCalendar, TaskStatus,
    schedule_with_ccpm, CCPMScheduleResult,
)
from fever_chart_system import FeverChartGenerator
from ccpm_execution_tracker import ProjectExecutionTracker, ProgressUpdate
from ccpm_sample_data import get_sample_tasks, get_sample_resources, get_sample_calendar

# Helper functions

def create_ccpm_schedule(
    tasks: List[Task],
    resources: List[Resource],
    calendar: ProjectCalendar,
    safety_factor: float,
    buffer_factor: float,
    delivery_date: Optional[int],
    verbose: bool
) -> 'CCPMScheduleResult':
    """Create CCPM schedule with buffers using the full pipeline."""

    tasks_dict = {t.id: t for t in tasks}
    resources_dict = {r.id: r for r in resources}

    if verbose:
        click.echo("Running full CCPM scheduling pipeline...")

    # Use the full CCPM pipeline from the module
    schedule_result = schedule_with_ccpm(
        tasks=tasks_dict,
        resources=resources_dict,
        project_calendar=calendar,
        max_day=delivery_date or 365, # Use delivery_date as max_day
        safety_factor=safety_factor,
        buffer_factor=buffer_factor
    )

    if verbose:
        click.echo("CCPM Schedule created successfully.")
        # Optionally print some stats from schedule_result.schedule_statistics

    return schedule_result

def generate_gantt_chart(schedule: 'CCPMScheduleResult', file_path: str) -> None:
    """Placeholder for Gantt chart generation."""
    raise NotImplementedError("Gantt chart generation is not yet implemented.")

def load_progress_updates(file_path: str) -> List[ProgressUpdate]:
    """Loads progress updates from a CSV file."""
    df = pd.read_csv(file_path)
    updates = []

    def to_optional_int(val: Any) -> Optional[int]:
        if pd.isna(val):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def to_optional_float(val: Any) -> Optional[float]:
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    for _, row in df.iterrows():
        status_str = row.get('status')
        status = TaskStatus[status_str.upper().replace(" ", "_")] if status_str and isinstance(status_str, str) else TaskStatus.NOT_STARTED

        update = ProgressUpdate(
            task_id=str(row['task_id']),
            status=status,
            actual_start=to_optional_int(row.get('actual_start')),
            actual_finish=to_optional_int(row.get('actual_finish')),
            percent_complete=to_optional_float(row.get('percent_complete'))
        )
        updates.append(update)
    return updates

def display_replan_summary(original_schedule: 'CCPMScheduleResult', new_schedule: 'CCPMScheduleResult', task_ids: List[str]) -> None:
    """Placeholder for displaying replan summary."""
    raise NotImplementedError("Replan summary display is not yet implemented.")

def display_detailed_schedule(schedule: 'CCPMScheduleResult') -> None:
    """Placeholder for displaying detailed schedule report."""
    raise NotImplementedError("Detailed schedule report is not yet implemented.")

def display_resource_report(schedule: 'CCPMScheduleResult') -> None:
    """Placeholder for displaying resource report."""
    raise NotImplementedError("Resource report is not yet implemented.")

def load_project_file(file_path: str) -> tuple[List[Task], List[Resource], ProjectCalendar]:
    """Load project data from various file formats"""
    path = Path(file_path)

    if path.suffix.lower() == '.csv':
        return load_from_csv(file_path)
    elif path.suffix.lower() == '.json':
        return load_from_json(file_path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return load_from_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def load_from_csv(file_path: str) -> tuple[List[Task], List[Resource], ProjectCalendar]:
    """Load project from CSV files"""
    # Implementation for CSV loading
    df = pd.read_csv(file_path)

    # Convert DataFrame to tasks
    tasks = []
    for _, row in df.iterrows():
        task = Task.from_dict(row.to_dict())
        tasks.append(task)

    # Create default resources and calendar
    resource_ids = set()
    for task in tasks:
        resource_ids.update(task.resources)

    resources = [
        Resource(resource_id=rid, name=rid)
        for rid in resource_ids
    ]

    calendar = ProjectCalendar()  # Default calendar

    return tasks, resources, calendar

def load_from_json(file_path: str) -> tuple[List[Task], List[Resource], ProjectCalendar]:
    """Load project from JSON file"""
    with open(file_path) as f:
        data = json.load(f)

    tasks = [Task.from_dict(task_data) for task_data in data.get('tasks', [])]
    resources = [Resource.from_dict(res_data) for res_data in data.get('resources', [])]
    calendar = ProjectCalendar.from_dict(data.get('calendar', {}))

    return tasks, resources, calendar

def load_from_excel(file_path: str) -> tuple[List[Task], List[Resource], ProjectCalendar]:
    """Load a complete project definition from an Excel file."""
    xls = pd.ExcelFile(file_path)

    # Load Tasks
    if 'Tasks' not in xls.sheet_names:
        raise ValueError("Excel file must contain a 'Tasks' sheet.")
    tasks_df = pd.read_excel(xls, sheet_name='Tasks')
    tasks = [Task.from_dict(row.to_dict()) for _, row in tasks_df.iterrows()]

    # Load Resources
    if 'Resources' in xls.sheet_names:
        resources_df = pd.read_excel(xls, sheet_name='Resources')
        resources = [Resource.from_dict(row.to_dict()) for _, row in resources_df.iterrows()]
    else:
        # Create default resources if sheet doesn't exist
        resource_ids = set()
        for task in tasks:
            resource_ids.update(task.resources)
        resources = [Resource(resource_id=rid, name=rid) for rid in resource_ids]

    # Load Project Calendar
    if 'ProjectCalendar' in xls.sheet_names:
        calendar_df = pd.read_excel(xls, sheet_name='ProjectCalendar')
        non_working_days = calendar_df['non_working_days'].tolist()
        calendar = ProjectCalendar(non_working_days=non_working_days)
    else:
        calendar = ProjectCalendar()  # Default calendar

    return tasks, resources, calendar

def save_schedule(schedule: 'CCPMScheduleResult', file_path: str) -> None:
    """Save schedule to JSON or Excel file based on extension."""
    if file_path.endswith('.xlsx'):
        # Save to Excel
        tasks_df = pd.DataFrame([t.to_dict() for t in schedule.tasks.values()])
        resources_df = pd.DataFrame([r.to_dict() for r in schedule.resources.values()])
        calendar_df = pd.DataFrame(schedule.project_calendar.to_dict()['non_working_days'], columns=['non_working_days'])

        with pd.ExcelWriter(file_path) as writer:
            tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
            resources_df.to_excel(writer, sheet_name='Resources', index=False)
            calendar_df.to_excel(writer, sheet_name='ProjectCalendar', index=False)
    else:
        # Save to JSON
        data = {
            'tasks': {tid: task.to_dict() for tid, task in schedule.tasks.items()},
            'resources': {rid: res.to_dict() for rid, res in schedule.resources.items()},
            'calendar': schedule.project_calendar.to_dict(),
            'critical_chain': schedule.critical_chain,
            'project_buffer': schedule.project_buffer.to_dict() if schedule.project_buffer else None,
            'feeding_buffers': {bid: buf.to_dict() for bid, buf in schedule.feeding_buffers.items()},
            'safety_tracker': {
                'removed_safety': schedule.safety_tracker.removed_safety,
                'chain_safety': schedule.safety_tracker.chain_safety,
            },
            'schedule_statistics': schedule.schedule_statistics
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

def load_schedule(file_path: str) -> 'CCPMScheduleResult':
    """Load schedule from a JSON file."""
    if not file_path.endswith('.json'):
        raise ValueError("Loading schedule from formats other than JSON is not supported.")

    with open(file_path) as f:
        data = json.load(f)

    from ccpm_module import SafetyTracker, ProjectBuffer, FeedingBuffer

    tasks = {tid: Task.from_dict(task_data) for tid, task_data in data['tasks'].items()}
    resources = {rid: Resource.from_dict(res_data) for rid, res_data in data.get('resources', {}).items()}
    calendar = ProjectCalendar.from_dict(data.get('calendar', {}))

    safety_tracker = SafetyTracker()
    if 'safety_tracker' in data:
        safety_tracker.removed_safety = data['safety_tracker'].get('removed_safety', {})
        safety_tracker.chain_safety = data['safety_tracker'].get('chain_safety', {})

    project_buffer = ProjectBuffer.from_dict(data['project_buffer']) if data.get('project_buffer') else None
    feeding_buffers = {bid: FeedingBuffer.from_dict(buf_data) for bid, buf_data in data.get('feeding_buffers', {}).items()}

    result = CCPMScheduleResult(
        tasks=tasks,
        resources=resources,
        project_calendar=calendar,
        critical_chain=data.get('critical_chain', []),
        project_buffer=project_buffer,
        feeding_buffers=feeding_buffers,
        safety_tracker=safety_tracker,
        schedule_statistics=data.get('schedule_statistics', {})
    )

    return result

def display_schedule_summary(schedule: 'CCPMScheduleResult') -> None:
    """Display schedule summary"""
    click.echo("\n" + "="*60)
    click.echo("CCPM SCHEDULE SUMMARY")
    click.echo("="*60)

    total_tasks = len(schedule.tasks)
    completed_tasks = sum(1 for t in schedule.tasks.values()
                         if t.execution_status == TaskStatus.COMPLETED)

    click.echo(f"Total Tasks: {total_tasks}")
    click.echo(f"Completed Tasks: {completed_tasks}")
    click.echo(f"Progress: {completed_tasks/total_tasks*100:.1f}%")

    if schedule.critical_chain:
        click.echo(f"Critical Chain Length: {len(schedule.critical_chain)} tasks")

    # Display project duration
    max_finish = max(
        (t.scheduled_finish or 0 for t in schedule.tasks.values()),
        default=0
    )
    click.echo(f"Project Duration: {max_finish} days")

    # Display safety statistics
    total_safety = sum(schedule.safety_tracker.removed_safety.values())
    click.echo(f"Total Safety Removed: {total_safety} days")

def display_buffer_status(tracker: ProjectExecutionTracker, current_date: int) -> None:
    """Displays the current status of all project buffers."""
    click.echo("\n" + "="*60)
    click.echo(f"BUFFER STATUS as of Day {current_date}")
    click.echo("="*60)

    statuses = tracker.get_buffer_statuses(current_date)

    if not statuses:
        click.echo("No buffers to display.")
        return

    for name, status in statuses.items():
        click.echo(f"\n--- {name} ({status['type']}) ---")
        click.echo(f"  Size: {status['size']} days")
        click.echo(f"  Planned Consumption: {status['planned_consumption']:.1f}%")
        click.echo(f"  Actual Consumption:  {status['actual_consumption']:.1f}%")

        # Add a simple status indicator
        if status['actual_consumption'] > 100:
            click.secho("  Status: CRITICAL (Overconsumed)", fg='red', bold=True)
        elif status['actual_consumption'] > status['planned_consumption'] + 20:
            click.secho("  Status: WARNING (Significantly behind plan)", fg='yellow')
        else:
            click.secho("  Status: ON TRACK", fg='green')

def display_critical_chain_report(schedule: 'CCPMScheduleResult') -> None:
    """Display critical chain analysis"""
    click.echo("\n" + "="*60)
    click.echo("CRITICAL CHAIN ANALYSIS")
    click.echo("="*60)

    if not schedule.critical_chain:
        click.echo("Critical chain not identified")
        return

    for task_id in schedule.critical_chain:
        task = schedule.tasks[task_id]
        click.echo(f"{task_id}: {task.name} ({task.duration} days)")

@click.group()
@click.version_option()
def cli():
    """
    CCPM Project Management Tool
    
    Critical Chain Project Management scheduling and execution tracking.
    """
    pass

@click.group()
def template():
    """
    Generate template files for project data.
    """
    pass

cli.add_command(template)

@template.command()
@click.option('--output', '-o', default='project_template.csv', help='Output file name.')
def csv(output):
    """Generate a sample project file in CSV format."""
    tasks = get_sample_tasks()

    # Convert tasks to a list of dictionaries for DataFrame creation
    task_data = []
    for task in tasks:
        task_data.append({
            'id': task.id,
            'name': task.name,
            'duration': task.duration,
            'predecessors': ",".join(task.predecessors),
            'resources': ",".join(task.resources),
            'description': task.description,
        })

    df = pd.DataFrame(task_data)
    df.to_csv(output, index=False)
    click.echo(f"Sample project file created: {output}")

@template.command(name='json')
@click.option('--output', '-o', default='project_template.json', help='Output file name.')
def json_template(output):
    """Generate a sample project file in JSON format."""
    tasks = get_sample_tasks()
    resources = get_sample_resources()
    calendar = get_sample_calendar()

    data = {
        "tasks": [task.to_dict() for task in tasks],
        "resources": [resource.to_dict() for resource in resources],
        "calendar": calendar.to_dict(),
    }

    with open(output, 'w') as f:
        json.dump(data, f, indent=2)
    click.echo(f"Sample project file created: {output}")

@template.command()
@click.option('--output', '-o', default='project_template.xlsx', help='Output file name.')
def excel(output):
    """Generate a sample project file in Excel format."""
    tasks = get_sample_tasks()
    resources = get_sample_resources()

    task_data = []
    for task in tasks:
        task_data.append({
            'id': task.id,
            'name': task.name,
            'duration': task.duration,
            'predecessors': ",".join(task.predecessors),
            'resources': ",".join(task.resources),
            'description': task.description,
        })
    tasks_df = pd.DataFrame(task_data)

    resource_data = []
    for resource in resources:
        resource_data.append({
            'id': resource.id,
            'name': resource.name,
            'capacity_per_day': resource.capacity_per_day,
        })
    resources_df = pd.DataFrame(resource_data)

    with pd.ExcelWriter(output) as writer:
        tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
        resources_df.to_excel(writer, sheet_name='Resources', index=False)

    click.echo(f"Sample project file created: {output}")

@cli.command()
@click.argument('project_file', type=click.Path(exists=True))
@click.option('--safety-factor', '-s', default=0.5, 
              help='Safety removal factor (0.0-1.0, default: 0.5)')
@click.option('--buffer-factor', '-b', default=0.5,
              help='Buffer sizing factor (0.0-1.0, default: 0.5)')
@click.option('--delivery-date', '-d', type=int,
              help='Target delivery date (project day)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for CCPM schedule (JSON format)')
@click.option('--gantt', type=click.Path(),
              help='Output file for Gantt chart (HTML format)')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def schedule(project_file: str, safety_factor: float, buffer_factor: float,
             delivery_date: Optional[int], output: Optional[str], 
             gantt: Optional[str], verbose: bool):
    """
    Create CCPM schedule from project file.
    
    PROJECT_FILE can be CSV, JSON, or Excel format.
    """
    try:
        # Validate parameters
        if not 0.0 <= safety_factor <= 1.0:
            raise click.BadParameter("Safety factor must be between 0.0 and 1.0")
        if not 0.0 <= buffer_factor <= 1.0:
            raise click.BadParameter("Buffer factor must be between 0.0 and 1.0")
        
        # Load project data
        click.echo(f"Loading project from {project_file}...")
        tasks, resources, calendar = load_project_file(project_file)
        
        if verbose:
            click.echo(f"Loaded {len(tasks)} tasks and {len(resources)} resources")
        
        # Create CCPM schedule
        click.echo("Creating CCPM schedule...")
        schedule_result = create_ccpm_schedule(
            tasks, resources, calendar,
            safety_factor=safety_factor,
            buffer_factor=buffer_factor,
            delivery_date=delivery_date,
            verbose=verbose
        )
        
        # Save results
        if output:
            save_schedule(schedule_result, output)
            click.echo(f"Schedule saved to {output}")
        
        if gantt:
            generate_gantt_chart(schedule_result, gantt)
            click.echo(f"Gantt chart saved to {gantt}")
        
        # Display summary
        display_schedule_summary(schedule_result)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('schedule_file', type=click.Path(exists=True))
@click.argument('progress_file', type=click.Path(exists=True))
@click.option('--current-date', '-d', type=int, required=True,
              help='Current project date')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for updated schedule')
@click.option('--fever-chart', type=click.Path(),
              help='Generate fever chart (HTML format)')
@click.option('--verbose', '-v', is_flag=True)
def update(schedule_file: str, progress_file: str, current_date: int,
           output: Optional[str], fever_chart: Optional[str], verbose: bool):
    """
    Update schedule with actual progress data.
    
    SCHEDULE_FILE: Previously created CCPM schedule (JSON)
    PROGRESS_FILE: Progress updates (CSV format)
    """
    try:
        # Load schedule and progress data
        click.echo(f"Loading schedule from {schedule_file}...")
        schedule_result = load_schedule(schedule_file)

        click.echo(f"Loading progress from {progress_file}...")
        progress_updates = load_progress_updates(progress_file)

        if verbose:
            click.echo(f"Processing {len(progress_updates)} progress updates")

        # Update schedule
        click.echo("Updating schedule with progress...")
        tracker = ProjectExecutionTracker(schedule_result)

        for p_update in progress_updates:
            tracker.update_task_progress(p_update)

        # Generate fever chart if requested
        if fever_chart:
            click.echo("Generating fever chart...")
            generator = FeverChartGenerator(schedule_result)
            chart_data = generator.generate_chart_data(
                start_date=0,
                end_date=current_date,
                progress_updates={t.id: t for t in schedule_result.tasks.values()}
            )
            
            html_chart = generator.generate_html_chart(chart_data)
            Path(fever_chart).write_text(html_chart)
            click.echo(f"Fever chart saved to {fever_chart}")
        
        # Save updated schedule
        if output:
            save_schedule(schedule_result, output)
            click.echo(f"Updated schedule saved to {output}")
        
        # Display buffer status
        display_buffer_status(tracker, current_date)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

# @cli.command()
# @click.argument('schedule_file', type=click.Path(exists=True))
# @click.argument('replan_tasks', help='Comma-separated task IDs to replan')
# @click.option('--current-date', '-d', type=int, required=True,
#               help='Current project date')
# @click.option('--output', '-o', type=click.Path(),
#               help='Output file for replanned schedule')
# @click.option('--verbose', '-v', is_flag=True)
# def replan(schedule_file: str, replan_tasks: str, current_date: int,
#            output: Optional[str], verbose: bool):
#     """
#     Replan specified tasks using original durations.
    
#     REPLAN_TASKS: Comma-separated list of task IDs to revert and replan
#     """
#     try:
#         # Load schedule
#         schedule_result = load_schedule(schedule_file)

#         # Parse task IDs
#         task_ids = [tid.strip() for tid in replan_tasks.split(',')]

#         if verbose:
#             click.echo(f"Replanning tasks: {task_ids}")

#         # Create replanner
#         from ccpm_module import CCPMReplanner
#         replanner = CCPMReplanner(schedule_result)

#         # Check which tasks can be replanned
#         replannable = replanner.identify_replannable_tasks()
#         invalid_tasks = [tid for tid in task_ids if tid not in replannable]

#         if invalid_tasks:
#             click.echo(f"Warning: Cannot replan tasks (already started): {invalid_tasks}")
#             task_ids = [tid for tid in task_ids if tid in replannable]

#         if not task_ids:
#             click.echo("No tasks available for replanning")
#             return

#         # Revert and replan
#         click.echo("Reverting task durations...")
#         replanner.revert_task_durations(task_ids)

#         click.echo("Creating new CCPM plan...")
#         new_schedule = replanner.replan_project(
#             current_date=current_date,
#             progress_updates={}
#         )

#         # Save replanned schedule
#         if output:
#             save_schedule(new_schedule, output)
#             click.echo(f"Replanned schedule saved to {output}")

#         # Display changes
#         display_replan_summary(schedule_result, new_schedule, task_ids)

#     except Exception as e:
#         click.echo(f"Error: {e}", err=True)
#         sys.exit(1)

@cli.command()
@click.argument('schedule_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./reports',
              help='Output directory for reports')
@click.option('--format', 'output_format', 
              type=click.Choice(['html', 'pdf', 'png']), default='html',
              help='Output format for fever chart')
@click.option('--start-date', type=int, default=0,
              help='Start date for chart')
@click.option('--end-date', type=int,
              help='End date for chart (default: project end)')
def fever_chart(schedule_file: str, output_dir: str, output_format: str,
                start_date: int, end_date: Optional[int]):
    """
    Generate fever chart report for buffer monitoring.
    """
    try:
        # Load schedule
        schedule_result = load_schedule(schedule_file)
        
        # Determine end date
        if end_date is None:
            end_date = max(
                task.scheduled_finish or 0 
                for task in schedule_result.tasks.values()
            )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate fever chart
        click.echo("Generating fever chart...")
        generator = FeverChartGenerator(schedule_result)
        chart_data = generator.generate_chart_data(
            start_date=start_date,
            end_date=end_date,
            progress_updates={t.id: t for t in schedule_result.tasks.values()}
        )
        
        # Save in requested format
        output_file = None
        if output_format == 'html':
            html_chart = generator.generate_html_chart(chart_data)
            output_file = output_path / 'fever_chart.html'
            output_file.write_text(html_chart)
        # elif output_format == 'png':
        #     # Generate static PNG chart
        #     png_data = generator.generate_static_chart(chart_data)
        #     output_file = output_path / 'fever_chart.png'
        #     output_file.write_bytes(png_data)
        # elif output_format == 'pdf':
        #     # Generate PDF report
        #     pdf_data = generator.generate_pdf_report(chart_data)
        #     output_file = output_path / 'fever_chart.pdf'
        #     output_file.write_bytes(pdf_data)
        
        click.echo(f"Fever chart saved to {output_file}")
        
        # Also save data as CSV for analysis
        csv_file = output_path / 'fever_chart_data.csv'
        chart_data.to_dataframe().to_csv(csv_file, index=False)
        click.echo(f"Chart data saved to {csv_file}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('schedule_file', type=click.Path(exists=True))
@click.option('--format', 'output_format',
              type=click.Choice(['summary', 'detailed', 'resources', 'critical-chain']),
              default='summary', help='Report type')
def report(schedule_file: str, output_format: str):
    """
    Generate various project reports.
    """
    try:
        schedule_result = load_schedule(schedule_file)
        
        if output_format == 'summary':
            display_schedule_summary(schedule_result)
        elif output_format == 'detailed':
            display_detailed_schedule(schedule_result)
        elif output_format == 'resources':
            display_resource_report(schedule_result)
        elif output_format == 'critical-chain':
            display_critical_chain_report(schedule_result)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)



if __name__ == '__main__':
    cli()