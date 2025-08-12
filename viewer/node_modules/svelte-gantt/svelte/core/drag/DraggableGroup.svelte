<script>import { setContext } from 'svelte';
import { writable } from 'svelte/store';
import { createEventDispatcher } from "svelte";
import { getContext } from 'svelte';
import { scrollIfOutOfBounds, setCursor, getRowAtPoint } from '../../utils/dom';
import { isDraggable, isResizable } from '../../utils/utils';
export let items = {};
let _taskIds = [];
const _active = writable(false);
const root = writable({});
$: $root = items;
const { taskStore, rowStore } = getContext('dataStore');
const gantt = getContext('gantt');
const { rowPadding } = getContext('options');
const { api, utils, columnService, selectionManager } = getContext('services');
const dispatcher = createEventDispatcher();
const selectedTasks = selectionManager._selectedTasks;
const context = {
    rootState: root,
    dragAllowed({ model }) {
        const row = $rowStore.entities[model.resourceId];
        return row && isDraggable(row.model) && isDraggable(model);
    },
    resizeAllowed({ model }) {
        const row = $rowStore.entities[model.resourceId];
        return model.type !== 'milestone' && row && isResizable(row.model) && isResizable(model);
    },
    off(taskId) {
        delete $root[taskId];
    },
    save(start, task, event) {
        // triggers onmousedown
        setCursor(event.dragging ? 'move' : 'e-resize');
        const { mouseEvent } = event;
        let tasks = [task];
        if (mouseEvent.ctrlKey) {
            for (const [taskId, isSelected] of Object.entries($selectedTasks)) {
                if (isSelected && taskId !== String(task.model.id)) {
                    tasks.push($taskStore.entities[taskId]);
                }
            }
        }
        _taskIds = [];
        for (const task of tasks) {
            _taskIds.push(task.model.id);
            $root[task.model.id] = {
                xDelta: task.left - start.x,
                yDelta: task.top - start.y,
                width: task.width,
                bWidth: start.width,
                widthDelta: task.width - start.width,
            };
        }
        dispatcher('itemsChange', { items: $root });
    },
    dropAll(event) {
        if (!event.dragging && !event.resizing) {
            return;
        }
        $_active = false;
        const changes = [];
        for (const taskId of _taskIds) {
            const state = $root[taskId];
            const task = $taskStore.entities[taskId];
            const isTarget = task.model.id === taskId;
            const change = onDropSingle({
                ...event,
                mouseEvent: {
                    ...event.mouseEvent,
                    clientX: event.mouseEvent.clientX + state.xDelta,
                    clientY: event.mouseEvent.clientY + state.yDelta,
                },
                x: event.x + state.xDelta,
                y: event.y + state.yDelta,
                width: Math.abs((isTarget ? event.width : state.bWidth) + state.widthDelta),
            }, task);
            changes.push({
                valid: change.valid,
                task: change.task,
                targetRow: change.targetRow,
                sourceRow: change.sourceRow,
                current: change.current,
                previous: change.previous,
            });
        }
        dispatcher('change', { changes });
        $root = {};
        _taskIds = [];
        dispatcher('itemsChange', { items: $root });
    },
    moveAll({ x, y, width, event }, task, state) {
        scrollIfOutOfBounds(event, gantt.mainContainer);
        if (state.dragging) {
            api.tasks.raise.move(task.model);
        }
        if (state.resizing) {
            api.tasks.raise.resize(task.model);
        }
        for (const taskId of _taskIds) {
            // somehow update tasks to (event.x + xDelta, event.y + yDelta)
            const rootState = $root[taskId];
            const isTarget = task.model.id === taskId;
            const event = {
                x: x != null ? x + rootState.xDelta : null,
                y: y != null ? y + rootState.yDelta : null,
                width: width != null ? Math.abs((isTarget ? width : rootState.bWidth) + rootState.widthDelta) : null, // pos.width + (width - pos.bWidth) // wDelta
            };
            $root[taskId] = {
                ...$root[taskId],
                ...state,
                x: event.x,
                y: event.y,
                width: event.width ?? $root[taskId]?.width,
            };
        }
    },
    setState(task, state) {
        $root[task.model.id] = {
            ...$root[task.model.id],
            ...state,
        };
    },
    mouseUp({ model }) {
        setCursor('default');
        api.tasks.raise.moveEnd(model);
    }
};
// honestly keep state in context provider.
function setState(state, id) {
    $root[id] = {
        ...$root[id],
        ...state,
    };
}
function onDropSingle(event, task) {
    //row switching
    const model = task.model;
    const sourceRow = $rowStore.entities[model.resourceId];
    let targetRow;
    if (event.dragging) {
        const rowId = getRowAtPoint(event.mouseEvent);
        const row = $rowStore.entities[rowId];
        if (row && isDraggable(row.model)) {
            targetRow = row;
        }
        // target row can be null
    }
    else {
        // dont know about this
        targetRow = $rowStore.entities[model.resourceId];
    }
    setState({
        dragging: false,
        resizing: false,
    }, model.id);
    setTimeout(() => {
        // sets state after it has been cleared, also resulting in a wierd render
        setState({
            ignoreClick: false,
        }, model.id);
    });
    const newFrom = utils.roundTo(columnService.getDateByPosition(event.x));
    const newTo = utils.roundTo(columnService.getDateByPosition(event.x + event.width));
    const newLeft = columnService.getPositionByDate(newFrom) | 0;
    const newRight = columnService.getPositionByDate(newTo) | 0;
    const left = newLeft;
    const width = newRight - newLeft;
    const top = $rowPadding + (targetRow?.y ?? 0);
    // get value of top from the layout
    const current = {
        left,
        top,
        width,
        from: newFrom,
        to: newTo,
    };
    const previous = {
        left: task.left,
        width: task.width,
        top: task.top,
        from: model.from,
        to: model.to,
    };
    return {
        valid: true,
        task,
        current,
        previous,
        dragging: event.dragging,
        resizing: event.resizing,
        sourceRow,
        targetRow,
    };
}
setContext('drag', context);
</script>

<slot {context}></slot>
