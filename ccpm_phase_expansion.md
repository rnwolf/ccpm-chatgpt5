# CCMP Tool - Detailed Phase 2 & 3 Plans

## Current Status Summary

**Phase 1 (COMPLETED):** Core infrastructure is stable with:
- Fixed `Resource` allocation tracking
- Consolidated calendar logic in `ProjectCalendar` class
- Basic unit testing with `test_ccmp.py`
- Working ASAP/ALAP scheduling with resource constraints
- Defensive error handling and cycle detection

## Phase 2: Verify Scheduling and Implement CCPM

### 2.1 Comprehensive Scheduling Algorithm Verification

#### 2.1.1 Enhanced Test Coverage

- **Expand `test_ccpm.py`** with comprehensive scheduling tests:
  ```python
  def test_simple_linear_schedule()
  def test_parallel_tasks_with_shared_resources()
  def test_complex_network_with_multiple_feeds()
  def test_resource_overallocation_handling()
  def test_alap_vs_asap_differences()
  ```

- **Create test fixtures** with known expected outcomes:
  - Simple 3-task linear chain (A→B→C)
  - Diamond network (A→B,C→D)
  - Complex feed structure with multiple merge points
  - Resource contention scenarios

#### 2.1.2 Scheduling Algorithm Validation

- **Verify ASAP Forward Pass:**
  - Test with various calendar patterns (weekends, holidays)
  - Validate early start calculations with multiple predecessors
  - Confirm working day arithmetic accuracy

- **Verify ALAP Backward Pass:**
  - Test late start calculations anchored to delivery dates
  - Validate resource-constrained scheduling logic
  - Test edge cases (zero duration tasks, single-day projects)

- **Cross-validation:**
  - Compare manual calculations vs. algorithmic results
  - Test schedule stability across multiple iterations
  - Validate that ASAP ≤ ALAP for all tasks

#### 2.1.3 Bug Fixes and Edge Cases

- **Resource allocation edge cases:**
  - Tasks with zero duration
  - Tasks requiring multiple units of same resource
  - Resource calendars with complex patterns

- **Calendar arithmetic edge cases:**
  - Projects spanning year boundaries
  - Leap year handling
  - Extended non-working periods

### 2.2 Critical Chain Identification Implementation

#### 2.2.1 Longest Path Algorithms

```python
def identify_critical_chain(
    tasks: Dict[str, Task],
    schedule: Dict[str, Tuple[int, int]],
    resources: Dict[str, Resource]
) -> List[str]:
    """
    Identify the critical chain considering both:
    1. Task dependencies (traditional critical path)
    2. Resource dependencies (CCPM innovation)
    """
```

**Implementation approach:**
- Build resource dependency graph (tasks sharing resources)
- Calculate float/slack for all tasks: `slack = alap_start - asap_start`
- Find longest path considering both task and resource constraints
- Mark tasks with `on_critical_chain = True`

#### 2.2.2 Resource Dependency Analysis

```python
def build_resource_dependency_graph(tasks: Dict[str, Task]) -> Dict[str, List[str]]:
    """Build graph showing which tasks are resource-dependent"""

def find_resource_constrained_paths(tasks: Dict[str, Task]) -> List[List[str]]:
    """Find all resource-constrained sequences"""
```

#### 2.2.3 Critical Chain Validation

- **Test critical chain detection:**
  - Verify critical chain is longest path through network
  - Confirm resource constraints properly considered
  - Test with various resource allocation patterns

- **Visual validation tools:**
  - Print critical chain path with durations
  - Show resource utilization along critical chain
  - Compare traditional CPM critical path vs. CCPM critical chain

### 2.3 Advanced Testing and Edge Case Handling

#### 2.3.1 Stress Testing

- **Large project networks:**
  - 50+ task projects
  - Complex resource sharing patterns
  - Multiple resource types with different calendars

- **Performance testing:**
  - Measure scheduling time complexity
  - Test memory usage with large datasets
  - Validate algorithm stability

#### 2.3.2 Enhanced Diagnostics

```python
def analyze_schedule_quality(tasks: Dict[str, Task]) -> Dict[str, Any]:
    """
    Provide comprehensive schedule analysis:
    - Resource utilization statistics
    - Critical chain analysis
    - Float/slack distribution
    - Potential bottlenecks
    """
```

## Phase 3: Implement Full CCPM Features

### 3.1 Duration Adjustment and Safety Removal

#### 3.1.1 Aggressive Duration Calculation

```python
def calculate_aggressive_durations(
    tasks: Dict[str, Task],
    safety_factor: float = 0.5
) -> Dict[str, int]:
    """
    Remove safety padding from individual task estimates.
    Default: reduce to 50% of original duration (aggressive but achievable).
    """
```

**Implementation details:**
- Store original durations in `task.original_duration`
- Calculate aggressive durations: `aggressive = original * (1 - safety_factor)`
- Track removed safety: `removed_safety = original - aggressive`
- Update task durations for scheduling

#### 3.1.2 Safety Tracking System

```python
class SafetyTracker:
    """Track safety time removed from tasks for buffer calculations"""
    def __init__(self):
        self.removed_safety: Dict[str, int] = {}
        self.chain_safety: Dict[str, int] = {}

    def calculate_chain_safety(self, chain_tasks: List[str]) -> int:
        """Sum removed safety for a chain of tasks"""
```

### 3.2 Buffer Calculation and Placement

#### 3.2.1 Project Buffer Implementation

```python
def calculate_project_buffer(critical_chain_tasks: List[str], safety_tracker: SafetyTracker) -> int:
    """
    Calculate project buffer size (typically 50% of removed critical chain safety)
    """

def insert_project_buffer(
    tasks: Dict[str, Task],
    buffer_size: int,
    delivery_date: Optional[int] = None
) -> ProjectBuffer:
    """Insert project buffer at end of critical chain"""
```

#### 3.2.2 Feeding Buffer Detection and Placement

Following the detailed logic in `CCPM_LOGIC_CHATGPT5.md`:

```python
def detect_feeding_chains(
    tasks: Dict[str, Task],
    critical_chain: List[str]
) -> Dict[str, List[str]]:
    """
    For each critical chain task, identify feeding chains that merge into it.
    Returns mapping: {critical_task_id: [feeding_chain_task_ids]}
    """

def detect_feeding_chain_merges(feeding_chains: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Detect where feeding chains merge before hitting critical chain.
    Handle complex merge scenarios.
    """

def calculate_feeding_buffers(
    feeding_chains: Dict[str, List[str]],
    safety_tracker: SafetyTracker
) -> Dict[str, FeedingBuffer]:
    """Calculate and create feeding buffer objects"""
```

**Buffer placement algorithm:**
1. **Precompute graph helpers** (topological order, reachability)
2. **Find critical chain** (resource-constrained longest path)
3. **For each critical task:** find feeding chain endpoints
4. **Group feeding tasks** into chains, detect merges
5. **Size buffers** at ~50% of removed feeding chain safety
6. **Place buffers** between feeding chains and critical chain

#### 3.2.3 Buffer Integration

```python
def integrate_buffers_into_schedule(
    tasks: Dict[str, Task],
    project_buffer: ProjectBuffer,
    feeding_buffers: Dict[str, FeedingBuffer]
) -> Dict[str, Task]:
    """
    Insert buffer tasks into project network with proper dependencies
    """
```

### 3.3 Complete CCPM Scheduling Pipeline

#### 3.3.1 Full CCPM Scheduler

```python
def schedule_with_ccpm(
    tasks: List[Task],
    resources: Dict[str, Resource],
    project_calendar: ProjectCalendar,
    delivery_date: Optional[int] = None,
    safety_factor: float = 0.5,
    buffer_factor: float = 0.5
) -> CCPMScheduleResult:
    """
    Complete CCPM scheduling pipeline:
    1. Initial ASAP/ALAP scheduling
    2. Critical chain identification
    3. Duration adjustment (safety removal)
    4. Buffer calculation and placement
    5. Final scheduling with buffers
    6. Validation and diagnostics
    """
```

#### 3.3.2 CCPM Results Container

```python
@dataclass
class CCPMScheduleResult:
    """Complete CCPM scheduling results"""
    tasks: Dict[str, Task]
    critical_chain: List[str]
    project_buffer: ProjectBuffer
    feeding_buffers: Dict[str, FeedingBuffer]
    safety_removed: Dict[str, int]
    schedule_statistics: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Export complete schedule to DataFrame"""

    def generate_gantt_data(self) -> Dict[str, Any]:
        """Prepare data for Gantt chart visualization"""
```

### 3.4 Buffer Management and Monitoring (Week 4)

#### 3.4.1 Buffer Status Tracking

```python
class BufferMonitor:
    """Track buffer consumption during project execution"""

    def __init__(self, buffers: Dict[str, Buffer]):
        self.buffers = buffers
        self.consumption_history: List[Dict[str, Any]] = []

    def update_buffer_status(self, task_completions: Dict[str, int]) -> Dict[str, str]:
        """
        Return buffer status zones:
        - Green (0-33% consumed)
        - Yellow (34-67% consumed)
        - Red (68-100% consumed)
        """

    def get_priority_actions(self) -> List[Dict[str, Any]]:
        """Recommend actions based on buffer status"""
```

#### 3.4.2 Schedule Update Integration

```python
def update_ccmp_schedule(
    original_schedule: CCPMScheduleResult,
    actual_progress: Dict[str, int],
    project_calendar: ProjectCalendar
) -> CCPMScheduleResult:
    """Update CCPM schedule based on actual task completions"""
```

### 3.5 Validation and Documentation (Week 5)

#### 3.5.1 Comprehensive Testing

- **End-to-end CCPM tests:**
  - Complete workflow from task definition to buffer monitoring
  - Validate buffer sizes and placements
  - Test schedule updates with actual progress

- **Regression testing:**
  - Ensure Phase 1 & 2 functionality remains intact
  - Performance benchmarks
  - Memory usage validation

#### 3.5.2 Enhanced Documentation

```python
# Add comprehensive docstrings following development guidelines
# Create usage examples for each major feature
# Document CCPM methodology and implementation decisions
```

## Phase 4: Enhancement Roadmap

### Multi-Capacity Resources

- Extend `Resource` class with `units_per_day > 1`
- Update allocation tracking for multi-unit resources
- Modify resource-constrained scheduling algorithms

### Visualization and Export

- Gantt chart generation (HTML/SVG output)
- Critical chain highlighting
- Buffer status visualization
- Export to MS Project format

### Advanced Features

- Resource learning curves
- Task splitting and preemption
- Probabilistic scheduling
- Portfolio-level CCPM

## Success Criteria

**Phase 2 Complete When:**
- All scheduling algorithms verified with comprehensive tests
- Critical chain identification working correctly
- Resource dependency analysis functional
- Schedule quality diagnostics implemented

**Phase 3 Complete When:**
- Full CCPM pipeline functional (safety removal → buffer calculation → final scheduling)
- Buffer monitoring system operational
- Complete test coverage for CCPM features
- Documentation and examples complete
- Performance meets requirements (100+ task projects in <5 seconds)

This phased approach ensures systematic development while maintaining code quality and following the established development guidelines in `GEMINI.md`.