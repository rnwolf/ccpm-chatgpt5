from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd

@dataclass
class FeverChartPoint:
    """Single point on a fever chart"""
    date: int
    buffer_id: str
    buffer_type: str  # "project" or "feeding"
    buffer_name: str
    planned_consumption: float  # Percentage of buffer that should be consumed
    actual_consumption: float   # Percentage actually consumed
    consumption_delta: float    # Difference (negative = ahead of plan)
    status_zone: str           # "green", "yellow", "red"
    trend: str                 # "improving", "stable", "deteriorating"
    
    @property
    def is_critical(self) -> bool:
        """True if buffer is in red zone or trending badly"""
        return self.status_zone == "red" or (
            self.status_zone == "yellow" and self.trend == "deteriorating"
        )

@dataclass
class BufferAlert:
    """Alert for buffer status issues"""
    date: int
    buffer_id: str
    alert_type: str  # "overconsumption", "trend_warning", "critical"
    message: str
    severity: int    # 1=info, 2=warning, 3=critical
    
@dataclass
class FeverChartData:
    """Complete fever chart dataset for a project"""
    project_name: str
    start_date: int
    end_date: int
    chart_points: List[FeverChartPoint] = field(default_factory=list)
    buffer_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    critical_events: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[BufferAlert] = field(default_factory=list)
    
    def get_buffer_points(self, buffer_id: str) -> List[FeverChartPoint]:
        """Get all points for a specific buffer"""
        return [p for p in self.chart_points if p.buffer_id == buffer_id]
    
    def get_points_by_date(self, date: int) -> List[FeverChartPoint]:
        """Get all buffer points for a specific date"""
        return [p for p in self.chart_points if p.date == date]
    
    def get_critical_buffers(self, date: int) -> List[FeverChartPoint]:
        """Get buffers in critical status at given date"""
        return [p for p in self.get_points_by_date(date) if p.is_critical]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis"""
        data = []
        for point in self.chart_points:
            data.append({
                'date': point.date,
                'buffer_id': point.buffer_id,
                'buffer_type': point.buffer_type,
                'buffer_name': point.buffer_name,
                'planned_consumption': point.planned_consumption,
                'actual_consumption': point.actual_consumption,
                'consumption_delta': point.consumption_delta,
                'status_zone': point.status_zone,
                'trend': point.trend
            })
        return pd.DataFrame(data)
    
    def to_json(self) -> str:
        """Export for web visualization"""
        return json.dumps({
            'project_name': self.project_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'chart_points': [
                {
                    'date': p.date,
                    'buffer_id': p.buffer_id,
                    'buffer_type': p.buffer_type,
                    'buffer_name': p.buffer_name,
                    'planned_consumption': p.planned_consumption,
                    'actual_consumption': p.actual_consumption,
                    'consumption_delta': p.consumption_delta,
                    'status_zone': p.status_zone,
                    'trend': p.trend
                }
                for p in self.chart_points
            ],
            'buffer_definitions': self.buffer_definitions,
            'critical_events': self.critical_events,
            'alerts': [
                {
                    'date': a.date,
                    'buffer_id': a.buffer_id,
                    'alert_type': a.alert_type,
                    'message': a.message,
                    'severity': a.severity
                }
                for a in self.alerts
            ]
        }, indent=2)

class BufferConsumptionCalculator:
    """Calculate buffer consumption based on project progress"""
    
    def __init__(self, ccmp_schedule: 'CCPMScheduleResult'):
        self.schedule = ccmp_schedule
        
    def calculate_planned_consumption(
        self, 
        buffer_id: str, 
        date: int
    ) -> float:
        """
        Calculate planned buffer consumption percentage at given date.
        Based on planned progress of feeding/critical chain tasks.
        """
        buffer = self._get_buffer(buffer_id)
        if not buffer:
            return 0.0
            
        if buffer.buffer_type == "project":
            return self._calculate_project_buffer_planned_consumption(buffer, date)
        else:
            return self._calculate_feeding_buffer_planned_consumption(buffer, date)
    
    def calculate_actual_consumption(
        self, 
        buffer_id: str, 
        date: int,
        progress_data: Dict[str, Task]
    ) -> float:
        """
        Calculate actual buffer consumption based on task progress.
        """
        buffer = self._get_buffer(buffer_id)
        if not buffer:
            return 0.0
            
        if buffer.buffer_type == "project":
            return self._calculate_project_buffer_actual_consumption(buffer, date, progress_data)
        else:
            return self._calculate_feeding_buffer_actual_consumption(buffer, date, progress_data)
    
    def _calculate_project_buffer_planned_consumption(
        self, 
        buffer: 'ProjectBuffer', 
        date: int
    ) -> float:
        """Calculate planned consumption for project buffer"""
        critical_chain_tasks = [
            task for task in self.schedule.tasks.values() 
            if task.on_critical_chain
        ]
        
        if not critical_chain_tasks:
            return 0.0
            
        # Calculate planned progress through critical chain
        total_critical_duration = sum(task.duration for task in critical_chain_tasks)
        planned_completion_by_date = 0
        
        for task in critical_chain_tasks:
            if task.scheduled_finish and task.scheduled_finish <= date:
                planned_completion_by_date += task.duration
            elif task.scheduled_start and task.scheduled_start <= date < task.scheduled_finish:
                # Partially complete
                days_elapsed = date - task.scheduled_start + 1
                planned_completion_by_date += min(days_elapsed, task.duration)
        
        planned_progress = planned_completion_by_date / total_critical_duration if total_critical_duration > 0 else 0
        return planned_progress * 100  # Convert to percentage
    
    def _calculate_project_buffer_actual_consumption(
        self, 
        buffer: 'ProjectBuffer', 
        date: int,
        progress_data: Dict[str, Task]
    ) -> float:
        """Calculate actual consumption for project buffer based on delays"""
        critical_chain_tasks = [
            task_id for task_id, task in self.schedule.tasks.items() 
            if task.on_critical_chain
        ]
        
        total_delay = 0
        for task_id in critical_chain_tasks:
            if task_id in progress_data:
                task = progress_data[task_id]
                delay = task.calculate_delay()
                total_delay += delay
        
        # Buffer consumption = delay / buffer size
        consumption_percentage = (total_delay / buffer.duration) * 100 if buffer.duration > 0 else 0
        return min(100, consumption_percentage)  # Cap at 100%
    
    def _calculate_feeding_buffer_planned_consumption(
        self, 
        buffer: 'FeedingBuffer', 
        date: int
    ) -> float:
        """Calculate planned consumption for feeding buffer"""
        # Find tasks that feed this buffer
        feeding_tasks = self._get_feeding_tasks(buffer)
        
        if not feeding_tasks:
            return 0.0
            
        total_feeding_duration = sum(task.duration for task in feeding_tasks)
        planned_completion_by_date = 0
        
        for task in feeding_tasks:
            if task.scheduled_finish and task.scheduled_finish <= date:
                planned_completion_by_date += task.duration
            elif task.scheduled_start and task.scheduled_start <= date < task.scheduled_finish:
                days_elapsed = date - task.scheduled_start + 1
                planned_completion_by_date += min(days_elapsed, task.duration)
        
        planned_progress = planned_completion_by_date / total_feeding_duration if total_feeding_duration > 0 else 0
        return planned_progress * 100
    
    def _calculate_feeding_buffer_actual_consumption(
        self, 
        buffer: 'FeedingBuffer', 
        date: int,
        progress_data: Dict[str, Task]
    ) -> float:
        """Calculate actual consumption for feeding buffer"""
        feeding_tasks = self._get_feeding_tasks(buffer)
        
        total_delay = 0
        for task in feeding_tasks:
            if task.id in progress_data:
                progress_task = progress_data[task.id]
                delay = progress_task.calculate_delay()
                total_delay += delay
        
        consumption_percentage = (total_delay / buffer.duration) * 100 if buffer.duration > 0 else 0
        return min(100, consumption_percentage)
    
    def _get_buffer(self, buffer_id: str) -> Optional['Buffer']:
        """Get buffer by ID from schedule"""
        # Implementation depends on how buffers are stored in CCPMScheduleResult
        pass
    
    def _get_feeding_tasks(self, buffer: 'FeedingBuffer') -> List['Task']:
        """Get tasks that feed into this buffer"""
        # Implementation depends on buffer-task relationships
        pass

class FeverChartGenerator:
    """Generate CCPM fever charts for buffer monitoring"""
    
    def __init__(self, ccpm_schedule: 'CCPMScheduleResult'):
        self.schedule = ccmp_schedule
        self.calculator = BufferConsumptionCalculator(ccmp_schedule)
        
    def generate_chart_data(
        self, 
        start_date: int,
        end_date: int,
        progress_updates: Dict[str, Task],
        include_projections: bool = True
    ) -> FeverChartData:
        """Generate complete fever chart data"""
        
        chart_data = FeverChartData(
            project_name=getattr(self.schedule, 'project_name', 'Project'),
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate points for each date and buffer
        for date in range(start_date, end_date + 1):
            for buffer_id, buffer in self._get_all_buffers().items():
                point = self._generate_chart_point(
                    buffer_id, buffer, date, progress_updates
                )
                chart_data.chart_points.append(point)
        
        # Add buffer definitions
        chart_data.buffer_definitions = self._get_buffer_definitions()
        
        # Identify critical events
        chart_data.critical_events = self._identify_critical_events(chart_data)
        
        # Generate alerts
        chart_data.alerts = self._generate_alerts(chart_data)
        
        return chart_data
    
    def _generate_chart_point(
        self, 
        buffer_id: str, 
        buffer: 'Buffer',
        date: int,
        progress_updates: Dict[str, Task]
    ) -> FeverChartPoint:
        """Generate single fever chart point"""
        
        planned = self.calculator.calculate_planned_consumption(buffer_id, date)
        actual = self.calculator.calculate_actual_consumption(buffer_id, date, progress_updates)
        delta = actual - planned
        
        # Determine status zone
        if actual <= 33:
            status_zone = "green"
        elif actual <= 67:
            status_zone = "yellow"
        else:
            status_zone = "red"
        
        # Determine trend (compare with previous few days)
        trend = self._calculate_trend(buffer_id, date, delta)
        
        return FeverChartPoint(
            date=date,
            buffer_id=buffer_id,
            buffer_type="project" if isinstance(buffer, ProjectBuffer) else "feeding",
            buffer_name=buffer.name,
            planned_consumption=planned,
            actual_consumption=actual,
            consumption_delta=delta,
            status_zone=status_zone,
            trend=trend
        )
    
    def _calculate_trend(self, buffer_id: str, date: int, current_delta: float) -> str:
        """Calculate trend based on recent consumption pattern"""
        # Look at last 3-5 days to determine trend
        # This is a simplified implementation
        if current_delta > 10:  # Significantly behind plan
            return "deteriorating"
        elif current_delta < -5:  # Ahead of plan
            return "improving"
        else:
            return "stable"
    
    def _get_all_buffers(self) -> Dict[str, 'Buffer']:
        """Get all buffers from schedule"""
        # Implementation depends on CCPMScheduleResult structure
        buffers = {}
        if hasattr(self.schedule, 'project_buffer'):
            buffers[self.schedule.project_buffer.id] = self.schedule.project_buffer
        if hasattr(self.schedule, 'feeding_buffers'):
            buffers.update(self.schedule.feeding_buffers)
        return buffers
    
    def _get_buffer_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get buffer metadata for chart"""
        definitions = {}
        for buffer_id, buffer in self._get_all_buffers().items():
            definitions[buffer_id] = {
                'name': buffer.name,
                'type': 'project' if isinstance(buffer, ProjectBuffer) else 'feeding',
                'duration': buffer.duration,
                'description': buffer.description
            }
        return definitions
    
    def _identify_critical_events(self, chart_data: FeverChartData) -> List[Dict[str, Any]]:
        """Identify critical events in the fever chart"""
        events = []
        
        # Find when buffers enter red zone
        for buffer_id in set(p.buffer_id for p in chart_data.chart_points):
            buffer_points = chart_data.get_buffer_points(buffer_id)
            
            red_entry = None
            for point in sorted(buffer_points, key=lambda p: p.date):
                if point.status_zone == "red" and red_entry is None:
                    red_entry = point.date
                    events.append({
                        'date': point.date,
                        'type': 'buffer_critical',
                        'buffer_id': buffer_id,
                        'description': f'Buffer {point.buffer_name} entered red zone'
                    })
                elif point.status_zone != "red" and red_entry is not None:
                    red_entry = None
        
        return events
    
    def _generate_alerts(self, chart_data: FeverChartData) -> List[BufferAlert]:
        """Generate alerts based on fever chart data"""
        alerts = []
        
        for point in chart_data.chart_points:
            if point.status_zone == "red":
                alerts.append(BufferAlert(
                    date=point.date,
                    buffer_id=point.buffer_id,
                    alert_type="critical",
                    message=f"Buffer {point.buffer_name} is in red zone ({point.actual_consumption:.1f}% consumed)",
                    severity=3
                ))
            elif point.status_zone == "yellow" and point.trend == "deteriorating":
                alerts.append(BufferAlert(
                    date=point.date,
                    buffer_id=point.buffer_id,
                    alert_type="trend_warning",
                    message=f"Buffer {point.buffer_name} consumption trending worse ({point.actual_consumption:.1f}%)",
                    severity=2
                ))
        
        return alerts
    
    def generate_html_chart(self, chart_data: FeverChartData) -> str:
        """Generate interactive HTML fever chart using Chart.js"""
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CCPM Fever Chart - {chart_data.project_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart-container {{ width: 100%; height: 600px; margin: 20px 0; }}
        .alerts {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 10px 0; }}
        .alert-critical {{ background: #f8d7da; border-color: #f5c6cb; }}
        .buffer-info {{ display: flex; gap: 20px; margin: 20px 0; }}
        .buffer-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
    </style>
</head>
<body>
    <h1>CCPM Fever Chart: {chart_data.project_name}</h1>
    
    <div class="buffer-info">
        {self._generate_buffer_cards(chart_data)}
    </div>
    
    <div class="chart-container">
        <canvas id="feverChart"></canvas>
    </div>
    
    <div class="alerts">
        <h3>Current Alerts</h3>
        {self._generate_alerts_html(chart_data)}
    </div>
    
    <script>
        {self._generate_chart_js(chart_data)}
    </script>
</body>
</html>"""
        
        return html_template
    
    def _generate_buffer_cards(self, chart_data: FeverChartData) -> str:
        """Generate HTML cards for buffer status"""
        cards = []
        latest_date = max(p.date for p in chart_data.chart_points)
        
        for buffer_id, buffer_def in chart_data.buffer_definitions.items():
            latest_points = [p for p in chart_data.chart_points 
                           if p.buffer_id == buffer_id and p.date == latest_date]
            
            if latest_points:
                point = latest_points[0]
                status_color = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#f8d7da"}[point.status_zone]
                
                cards.append(f"""
                <div class="buffer-card" style="background-color: {status_color}">
                    <h4>{buffer_def['name']}</h4>
                    <p>Type: {buffer_def['type'].title()}</p>
                    <p>Consumption: {point.actual_consumption:.1f}%</p>
                    <p>Status: {point.status_zone.title()}</p>
                    <p>Trend: {point.trend.title()}</p>
                </div>""")
        
        return "".join(cards)
    
    def _generate_alerts_html(self, chart_data: FeverChartData) -> str:
        """Generate HTML for current alerts"""
        if not chart_data.alerts:
            return "<p>No current alerts</p>"
        
        latest_date = max(p.date for p in chart_data.chart_points)
        current_alerts = [a for a in chart_data.alerts if a.date == latest_date]
        
        if not current_alerts:
            return "<p>No current alerts</p>"
        
        alerts_html = []
        for alert in current_alerts:
            css_class = "alert-critical" if alert.severity >= 3 else ""
            alerts_html.append(f"""
            <div class="alert {css_class}">
                <strong>{alert.alert_type.replace('_', ' ').title()}:</strong> {alert.message}
            </div>""")
        
        return "".join(alerts_html)
    
    def _generate_chart_js(self, chart_data: FeverChartData) -> str:
        """Generate Chart.js configuration"""
        
        # Prepare data for Chart.js
        datasets = []
        buffer_ids = list(set(p.buffer_id for p in chart_data.chart_points))
        
        for buffer_id in buffer_ids:
            buffer_points = sorted(
                [p for p in chart_data.chart_points if p.buffer_id == buffer_id],
                key=lambda p: p.date
            )
            
            datasets.append({
                'label': f"{buffer_points[0].buffer_name} (Planned)",
                'data': [{'x': p.date, 'y': p.planned_consumption} for p in buffer_points],
                'borderColor': 'blue',
                'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                'borderDash': [5, 5],
                'fill': False
            })
            
            datasets.append({
                'label': f"{buffer_points[0].buffer_name} (Actual)",
                'data': [{'x': p.date, 'y': p.actual_consumption} for p in buffer_points],
                'borderColor': 'red',
                'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                'fill': False
            })
        
        chart_config = f"""
        const ctx = document.getElementById('feverChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                datasets: {json.dumps(datasets)}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'linear',
                        position: 'bottom',
                        title: {{
                            display: true,
                            text: 'Project Day'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Buffer Consumption (%)'
                        }},
                        min: 0,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'CCPM Buffer Consumption Over Time'
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    annotation: {{
                        annotations: {{
                            yellowLine: {{
                                type: 'line',
                                yMin: 33,
                                yMax: 33,
                                borderColor: 'orange',
                                borderWidth: 2,
                                borderDash: [10, 5],
                                label: {{
                                    display: true,
                                    content: 'Yellow Zone (33%)',
                                    position: 'end'
                                }}
                            }},
                            redLine: {{
                                type: 'line',
                                yMin: 67,
                                yMax: 67,
                                borderColor: 'red',
                                borderWidth: 2,
                                borderDash: [10, 5],
                                label: {{
                                    display: true,
                                    content: 'Red Zone (67%)',
                                    position: 'end'
                                }}
                            }}
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }},
                tooltips: {{
                    mode: 'index',
                    intersect: false,
                    callbacks: {{
                        label: function(context) {{
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                        }}
                    }}
                }}
            }}
        }});"""
        
        return chart_config