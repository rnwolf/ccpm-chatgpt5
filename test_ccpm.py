import pytest
from ccpm_module import ProjectCalendar, Resource

def test_project_calendar_initialization():
    calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    assert not calendar.is_working_day(5)
    assert not calendar.is_working_day(6)
    assert calendar.is_working_day(7)

def test_project_calendar_next_working_day():
    calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    assert calendar.next_working_day(4) == 4
    assert calendar.next_working_day(5) == 7
    assert calendar.next_working_day(6) == 7

def test_project_calendar_prev_working_day():
    calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    assert calendar.prev_working_day(7) == 7
    assert calendar.prev_working_day(6) == 4
    assert calendar.prev_working_day(5) == 4

def test_project_calendar_add_working_days():
    calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    # Starts on Mon (day 0), duration 4 days -> finishes on Thu (day 3). Returns day after (4).
    assert calendar.add_working_days(0, 4) == 4
    # Starts on Thu (day 3), duration 3 days. Working days are 3, 4, 7. Finishes on day 7. Returns 8.
    assert calendar.add_working_days(3, 3) == 8

def test_project_calendar_subtract_working_days():
    calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    # Ends on Thu (day 3), duration 4 days -> starts on Mon (day 0).
    assert calendar.subtract_working_days(3, 4) == 0
    # Ends on Tue (day 8), duration 3 days -> Tue, Mon, Fri. Starts on day 4.
    assert calendar.subtract_working_days(8, 3) == 4

def test_resource_initialization():
    res = Resource(resource_id="R1", name="Alice", non_working_days=[7])
    assert res.id == "R1"
    assert res.name == "Alice"
    assert 7 in res.non_working_days

def test_resource_is_available():
    project_calendar = ProjectCalendar(non_working_days=[5, 6, 12, 13])
    # Alice is off on day 7
    res = Resource(resource_id="R1", name="Alice", non_working_days=[7])

    assert res.is_available(4, project_calendar)  # Thursday, available
    assert not res.is_available(5, project_calendar) # Project non-working day
    assert not res.is_available(7, project_calendar) # Resource non-working day
    assert res.is_available(8, project_calendar) # Monday, available
