import pytest
from ccpm_module import (
    Task,
    TaskStatus,
    ProjectCalendar,
    Resource,
    schedule_tasks,
    identify_critical_chain,
    SafetyTracker,
    calculate_project_buffer,
    calculate_feeding_buffers,
    detect_feeding_chains,
    schedule_with_ccpm,
)
from ccpm_execution_tracker import ProjectExecutionTracker, ProgressUpdate

def test_original_duration_preservation():
    """Test that original durations are preserved for replanning and that related fields are updated correctly."""
    task = Task("T1", "Test", 10)

    # Check initial state
    assert task.duration == 10
    assert task.original_duration is None
    assert task.ccpm_duration is None
    assert task.safety_removed == 0

    # Apply CCPM safety reduction
    task.apply_ccpm_safety_reduction(0.5)

    # Check state after reduction
    assert task.original_duration == 10
    assert task.ccpm_duration == 5
    assert task.duration == 5  # Duration should be updated to the aggressive estimate
    assert task.safety_removed == 5

    # Test reversion
    assert task.revert_to_original_duration() is True

    # Check state after reversion
    assert task.duration == 10
    assert task.original_duration is None
    assert task.ccpm_duration is None
    assert task.safety_removed == 0

    # Test that reversion fails if no original duration is set
    assert task.revert_to_original_duration() is False


def test_critical_chain_identification_with_resource_contention():
    """
    Tests that the critical chain correctly identifies the longest path
    considering both precedence and resource constraints.
    """
    calendar = ProjectCalendar()
    resources = {
        "R1": Resource("R1", "Worker"),
    }
    tasks = {
        "A": Task("A", "Setup", 2, predecessors=[]),
        "B": Task("B", "Short Path", 3, resources=["R1"], predecessors=["A"]),
        "C": Task("C", "Long Path", 5, resources=["R1"], predecessors=["A"]),
        "D": Task("D", "Cleanup", 2, predecessors=["B", "C"]),
    }

    # Manually compute successors for this test, as schedule_tasks does this internally
    # but we want to be explicit for the test setup.
    tasks["A"].successors = ["B", "C"]
    tasks["B"].successors = ["D"]
    tasks["C"].successors = ["D"]

    # Schedule the project to create resource dependencies
    schedule = schedule_tasks(tasks, resources, calendar, max_day=100)

    # Identify the critical chain based on the ALAP schedule
    critical_chain = identify_critical_chain(tasks, schedule["ALAP"])

    # The longest path is through C. Due to resource contention, B and C are serialized.
    # The critical chain must include the resource-constrained sequence.
    alap_b_start = schedule["ALAP"]["B"][0]
    alap_c_start = schedule["ALAP"]["C"][0]

    # The ALAP scheduler will place tasks as late as possible. The one that can
    # finish later without pushing the project out will be scheduled first (in reverse).
    # In this case, the sequence depends on the scheduler's tie-breaking.
    # The critical path is the one that results from this scheduling.
    if alap_b_start < alap_c_start:
        expected_chain = ["A", "B", "C", "D"]
    else:
        expected_chain = ["A", "C", "B", "D"]

    assert critical_chain == expected_chain
    assert tasks["A"].on_critical_chain
    assert tasks["B"].on_critical_chain
    assert tasks["C"].on_critical_chain
    assert tasks["D"].on_critical_chain

    # Add a task that is truly independent to ensure it's not on the chain.
    tasks["E"] = Task("E", "Independent Task", 1)
    # Re-run the full pipeline to ensure E is handled
    schedule_with_e = schedule_tasks(tasks, resources, calendar, max_day=100)
    critical_chain_with_e = identify_critical_chain(tasks, schedule_with_e["ALAP"])

    assert not tasks["E"].on_critical_chain
    # The original chain should still be critical
    for tid in expected_chain:
        assert tasks[tid].on_critical_chain


def test_buffer_calculations_from_safety_removed():
    """
    Tests that project and feeding buffers are correctly sized based on the
    amount of safety removed from tasks.
    """
    # 1. Setup tasks and apply safety reduction
    tasks = {
        "CC1": Task("CC1", "Critical 1", 10),
        "CC2": Task("CC2", "Critical 2", 20),
        "FC1": Task("FC1", "Feeding 1", 8),
        "FC2": Task("FC2", "Feeding 2", 12),
    }

    for task in tasks.values():
        task.apply_ccpm_safety_reduction(0.5)

    safety_tracker = SafetyTracker()
    for tid, task in tasks.items():
        safety_tracker.removed_safety[tid] = task.safety_removed

    # 2. Test Project Buffer calculation
    critical_chain = ["CC1", "CC2"]
    # Safety removed from CC1 is 5 (10 * 0.5 = 5), from CC2 is 10 (20 * 0.5 = 10). Total = 15.
    # The apply_ccpm_safety_reduction method uses int(), so safety is 5 and 10.
    # Project buffer is 50% of total safety: 15 * 0.5 = 7.5.
    # The calculate_project_buffer function uses round(), so round(7.5) = 8.
    project_buffer_size = calculate_project_buffer(critical_chain, safety_tracker, buffer_factor=0.5)
    assert project_buffer_size == 8

    # 3. Test Feeding Buffer calculation
    feeding_chains = {
        "CC2": ["FC1", "FC2"]  # Feeding chain merges into CC2
    }
    # Safety removed from FC1 is 4 (8 * 0.5), from FC2 is 6 (12 * 0.5). Total = 10.
    # Feeding buffer is 50% of total safety: 10 * 0.5 = 5. round(5) = 5.
    feeding_buffers = calculate_feeding_buffers(feeding_chains, safety_tracker, buffer_factor=0.5)

    assert "FeedingBuffer-CC2" in feeding_buffers
    assert feeding_buffers["FeedingBuffer-CC2"].duration == 5


def test_complex_feeding_chain_detection():
    """
    Tests the detection of feeding chains in a more complex project structure.

    Scenario:
    - Critical Chain: A -> B -> C
    - Feeding Chain 1: F1 -> F2 (merges at B)
    - Feeding Chain 2: G1 -> C (merges at C)
    - Feeding Chain 3: H1 -> F2 (a sub-feed that merges into another feeding chain)
    """
    tasks = {
        "A": Task("A", "A", 1, predecessors=[]),
        "B": Task("B", "B", 1, predecessors=["A", "F2"]),
        "C": Task("C", "C", 1, predecessors=["B", "G1"]),
        "F1": Task("F1", "F1", 1, predecessors=[]),
        "F2": Task("F2", "F2", 1, predecessors=["F1", "H1"]),
        "G1": Task("G1", "G1", 1, predecessors=[]),
        "H1": Task("H1", "H1", 1, predecessors=[]),
    }

    # Set successors to build the graph for the detection logic
    tasks["A"].successors = ["B"]
    tasks["F1"].successors = ["F2"]
    tasks["H1"].successors = ["F2"]
    tasks["F2"].successors = ["B"]
    tasks["G1"].successors = ["C"]
    tasks["B"].successors = ["C"]

    critical_chain = ["A", "B", "C"]
    for tid in critical_chain:
        tasks[tid].on_critical_chain = True

    feeding_chains = detect_feeding_chains(tasks, critical_chain)

    assert "B" in feeding_chains, "Should detect feeding chain merging at B"
    assert set(feeding_chains["B"]) == {"F1", "F2", "H1"}, "Should find all tasks in the chain feeding B"

    assert "C" in feeding_chains, "Should detect feeding chain merging at C"
    assert set(feeding_chains["C"]) == {"G1"}, "Should find all tasks in the chain feeding C"

    assert "A" not in feeding_chains, "Critical chain tasks should not be merge points of feeding chains into themselves"


def test_execution_tracking_and_buffer_consumption():
    """
    Tests the full execution tracking loop, including progress updates
    and buffer consumption calculation.
    """
    # 1. Setup a scheduled project
    calendar = ProjectCalendar()
    resources = {"R1": Resource("R1", "Worker")}
    tasks = {
        "A": Task("A", "Critical Task 1", 10, resources=["R1"]),
        "B": Task("B", "Critical Task 2", 10, resources=["R1"], predecessors=["A"]),
    }

    # schedule_with_ccpm will handle successor computation
    schedule_result = schedule_with_ccpm(tasks, resources, calendar, max_day=100)

    # Initial state checks
    assert schedule_result.critical_chain == ["A", "B"]
    assert schedule_result.project_buffer is not None
    assert schedule_result.project_buffer.duration > 0

    # 2. Create a tracker
    tracker = ProjectExecutionTracker(schedule_result)

    # 3. Simulate a progress update for Task A with a delay
    # Aggressive duration for A is 10 * 0.5 = 5 days.
    # Its scheduled finish would be around day 4 (if start is 0).
    # Let's say it actually finishes on day 11 (duration of 12 days).
    task_a_scheduled_finish = schedule_result.tasks["A"].scheduled_finish
    assert task_a_scheduled_finish is not None

    update = ProgressUpdate(
        task_id="A",
        status=TaskStatus.COMPLETED,
        actual_start=0,
        actual_finish=11
    )
    tracker.update_task_progress(update)

    # Verify task A's status is updated
    assert tasks["A"].execution_status == TaskStatus.COMPLETED
    assert tasks["A"].actual_finish == 11

    # 4. Check buffer consumption
    delay = tasks["A"].calculate_delay()
    assert delay == 7 # (11 - 0 + 1) - 5 = 7

    statuses = tracker.get_buffer_statuses(current_date=12)
    project_buffer_status = statuses.get("Project Buffer")
    assert project_buffer_status is not None

    # Project buffer size is round((5+5)*0.5) = 5
    project_buffer_size = schedule_result.project_buffer.duration
    assert project_buffer_size == 5

    # Expected consumption is (7 / 5) * 100 = 140, which is capped at 100.
    expected_consumption = 100.0

    assert abs(project_buffer_status["actual_consumption"] - expected_consumption) < 0.1
