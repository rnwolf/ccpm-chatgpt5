import os
import pytest
import pandas as pd
from click.testing import CliRunner
from ccpm_cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_template_command_exists(runner):
    """Test that the 'template' command exists."""
    result = runner.invoke(cli, ['template', '--help'])
    assert result.exit_code == 0
    assert 'Generate template files for project data' in result.output

def test_template_csv_command(runner):
    """Test the 'template csv' command."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['template', 'csv'])
        assert result.exit_code == 0
        assert os.path.exists('project_template.csv')
        # We can add more assertions here to check the content of the file

def test_template_json_command(runner):
    """Test the 'template json' command."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['template', 'json'])
        assert result.exit_code == 0
        assert os.path.exists('project_template.json')
        # We can add more assertions here to check the content of the file

def test_template_excel_command(runner):
    """Test the 'template excel' command."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['template', 'excel'])
        assert result.exit_code == 0
        assert os.path.exists('project_template.xlsx')
        # We can add more assertions here to check the content of the file

def test_schedule_save_to_excel(runner):
    """Test that the schedule command can save its output to an Excel file."""
    with runner.isolated_filesystem():
        # First, create a project file to schedule
        project_file = 'test_project.csv'
        with open(project_file, 'w') as f:
            f.write("id,name,duration,predecessors,resources\n")
            f.write("A,Task A,5,,R1\n")
            f.write("B,Task B,5,A,R1\n")

        output_file = 'output_schedule.xlsx'
        result = runner.invoke(cli, ['schedule', project_file, '--output', output_file])

        assert result.exit_code == 0
        assert os.path.exists(output_file)

        # Verify the content of the Excel file
        xls = pd.ExcelFile(output_file)
        assert 'Tasks' in xls.sheet_names
        assert 'Resources' in xls.sheet_names
        assert 'ProjectCalendar' in xls.sheet_names

        tasks_df = pd.read_excel(xls, sheet_name='Tasks')
        assert len(tasks_df) > 0 # Should have tasks and buffers

        resources_df = pd.read_excel(xls, sheet_name='Resources')
        assert len(resources_df) > 0
