import { setContext, getContext } from "svelte";
function controller() {
    const listeners = [];
    function raise(...params) {
        for (const listener of listeners) {
            listener(params);
        }
    }
    ;
    function on(handler) {
        listeners.push(handler);
        const removeListener = () => {
            const index = listeners.indexOf(handler);
            listeners.splice(index, 1);
        };
        return removeListener;
    }
    ;
    return [on, raise];
}
function feature(events) {
    const result = { on: {}, raise: {} };
    for (const event in events) {
        const [on, raise] = events[event];
        result.on[event] = on;
        result.raise[event] = raise;
    }
    return result;
}
const contextKey = {};
export function provideGanttApi() {
    return setContext(contextKey, createGanttApi());
}
function createGanttApi() {
    return {
        tasks: feature({
            move: controller(),
            resize: controller(),
            select: controller(),
            switchRow: controller(),
            moveEnd: controller(),
            change: controller(),
            changed: controller(),
            dblclicked: controller(),
        }),
        gantt: feature({
            viewChanged: controller(),
            dateSelected: controller(),
        }),
        timeranges: feature({
            clicked: controller(),
            resized: controller(),
            changed: controller(),
        }),
    };
}
export function useGanttApi() {
    return getContext(contextKey);
}
