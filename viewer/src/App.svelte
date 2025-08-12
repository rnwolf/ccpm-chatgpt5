<script>
  import { SvelteGantt } from 'svelte-gantt/svelte';
  import { MomentSvelteGanttDateAdapter } from './dateAdapter.js';
  import { tasks, rows, timeRanges } from './data.js';

  const dateAdapter = new MomentSvelteGanttDateAdapter();

  const options = {
    dateAdapter,
    rows,
    tasks,
    timeRanges,
    from: new Date(2024, 5, 10),
    to: new Date(2024, 5, 30),
    fitWidth: true,
    rowHeight: 40
  };
</script>

<SvelteGantt {...options} />