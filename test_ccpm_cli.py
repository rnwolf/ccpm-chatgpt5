import os
import pytest
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
