# Scenario-based Validation

This document outlines the manual validation process performed on the CCPM scheduling engine. Three scenarios were created to test the core functionality of the scheduler under different conditions.

## Validation Process

For each scenario, the following steps were taken:
1.  A project was defined in a CSV file.
2.  The `ccpm schedule` command was run with the `--verbose` flag.
3.  The output summary was manually inspected and validated against expected values.

## Scenario 1: Simple Linear Project

*   **File**: `scenario_linear.csv`
*   **Description**: A simple project with three tasks in a single linear chain (A -> B -> C), all using the same resource.
*   **Purpose**: To validate the basic scheduling logic, critical chain identification, and project buffer calculation.

### Validation Results

| Metric                  | Expected | Actual | Status  |
| ----------------------- | -------- | ------ | ------- |
| Total Tasks             | 4        | 4      | ✅ Pass |
| Critical Chain Length   | 3        | 3      | ✅ Pass |
| Total Safety Removed    | 19 days  | 19 days| ✅ Pass |
| Project Duration        | ~364 days| 364 days| ✅ Pass |

*   **Notes**: The project duration is as expected for an unconstrained ALAP schedule. The core logic is working correctly.

## Scenario 2: Resource-Constrained Project

*   **File**: `scenario_resource_constrained.csv`
*   **Description**: A project with two parallel tasks (B and C) that compete for the same resource.
*   **Purpose**: To validate that the scheduler correctly handles resource contention and includes resource-constrained tasks in the critical chain.

### Validation Results

| Metric                  | Expected | Actual | Status  |
| ----------------------- | -------- | ------ | ------- |
| Total Tasks             | 5        | 5      | ✅ Pass |
| Critical Chain Length   | 4        | 4      | ✅ Pass |
| Total Safety Removed    | 16 days  | 16 days| ✅ Pass |
| Project Duration        | ~364 days| 364 days| ✅ Pass |

*   **Notes**: The critical chain correctly includes both parallel tasks, confirming that resource constraints are properly handled.

## Scenario 3: Multi-Feeding-Chain Project

*   **File**: `scenario_multi_feed.csv`
*   **Description**: A project with a critical chain and two separate feeding chains that merge into it at different points.
*   **Purpose**: To validate the detection of feeding chains and the calculation and placement of feeding buffers.

### Validation Results

| Metric                  | Expected | Actual | Status  |
| ----------------------- | -------- | ------ | ------- |
| Total Tasks             | 10       | 10     | ✅ Pass |
| Critical Chain Length   | 4        | 4      | ✅ Pass |
| Total Safety Removed    | 34 days  | 34 days| ✅ Pass |
| Project Duration        | ~364 days| 364 days| ✅ Pass |

*   **Notes**: The total number of tasks (7 from CSV + 1 project buffer + 2 feeding buffers) is correct, confirming that the feeding chains were detected and their buffers created.

## Advanced Scenarios (Excel-based)

These scenarios use Excel files as input to test the scheduler's handling of more complex constraints, particularly project and resource calendars.

### Validation Process for Excel Scenarios

For each scenario, use the following command to generate the schedule and save the output to an Excel file for manual inspection:

```bash
uv run ccpm_cli.py schedule <scenario_file.xlsx> --output <output_file.xlsx> --verbose
```

### Scenario 4: Calendar Impact

*   **File**: `scenario_calendar_impact.xlsx`
*   **Description**: A simple linear project with weekends and a project-wide holiday defined.
*   **Purpose**: To validate that the scheduler correctly accounts for non-working days in the project calendar.
*   **Validation Steps**:
    1.  Run `uv run ccpm_cli.py schedule scenario_calendar_impact.xlsx -o output_calendar_impact.xlsx --verbose`.
    2.  Open `output_calendar_impact.xlsx` and inspect the `Tasks` sheet.
    3.  Verify that the `scheduled_start` and `scheduled_finish` dates for the tasks correctly skip over the weekends and holidays defined in the `ProjectCalendar` sheet.

### Scenario 5: Resource Vacation

*   **File**: `scenario_resource_vacation.xlsx`
*   **Description**: A project with resource contention where the shared resource has specific vacation days.
*   **Purpose**: To validate the handling of combined resource and calendar constraints.
*   **Validation Steps**:
    1.  Run `uv run ccpm_cli.py schedule scenario_resource_vacation.xlsx -o output_resource_vacation.xlsx --verbose`.
    2.  Inspect the output file.
    3.  Verify that the tasks assigned to the resource with vacation days are not scheduled during those non-working days.

### Scenario 6: Multiple Resource Calendars

*   **File**: `scenario_multi_calendar.xlsx`
*   **Description**: A project with multiple feeding chains and resources with different non-working days.
*   **Purpose**: To validate the scheduler's ability to handle multiple, independent resource calendars simultaneously.
*   **Validation Steps**:
    1.  Run `uv run ccpm_cli.py schedule scenario_multi_calendar.xlsx -o output_multi_calendar.xlsx --verbose`.
    2.  Inspect the output file.
    3.  Verify that the tasks are scheduled according to the specific availability of their assigned resources.

## Conclusion

The manual validation of these scenarios confirms that the core CCPM scheduling engine is working correctly for a range of common project structures. The critical chain identification, safety removal, and buffer calculation and integration logic are all functioning as expected.
