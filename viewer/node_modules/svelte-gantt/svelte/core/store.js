import { writable, derived } from 'svelte/store';
function createEntityStore() {
    const { subscribe, set, update } = writable({ ids: [], entities: {} });
    return {
        set,
        _update: update,
        subscribe,
        add: (item) => update(({ ids, entities }) => ({
            ids: [...ids, item.model.id],
            entities: {
                ...entities,
                [item.model.id]: item
            }
        })),
        delete: (id) => update(state => {
            const { [id]: _, ...entities } = state.entities;
            return {
                ids: state.ids.filter(i => i !== id),
                entities
            };
        }),
        deleteAll: (ids) => update(state => {
            const entities = { ...state.entities };
            const idSet = new Set(ids);
            for (let i = 0; i < state.ids.length; i++) {
                if (idSet.has(state.ids[i])) {
                    delete entities[state.ids[i]];
                }
            }
            return {
                ids: state.ids.filter(i => !idSet.has(i)),
                entities
            };
        }),
        update: (item) => update(({ ids, entities }) => ({
            ids,
            entities: {
                ...entities,
                [item.model.id]: item
            }
        })),
        upsert: (item) => update(({ ids, entities }) => {
            const hasIndex = ids.indexOf(item.model.id) !== -1;
            return {
                ids: hasIndex ? ids : [...ids, item.model.id],
                entities: {
                    ...entities,
                    [item.model.id]: item
                }
            };
        }),
        upsertAll: (items) => update(state => {
            const entities = { ...state.entities };
            const ids = [...state.ids];
            for (let i = 0; i < items.length; i++) {
                if (ids.indexOf(items[i].model.id) === -1) {
                    ids.push(items[i].model.id);
                }
                entities[items[i].model.id] = items[i];
            }
            return {
                ids,
                entities
            };
        }),
        addAll: (items) => {
            const ids = [];
            const entities = {};
            for (let i = 0; i < items.length; i++) {
                ids.push(items[i].model.id);
                entities[items[i].model.id] = items[i];
            }
            set({ ids, entities });
        },
        refresh: () => update(store => ({ ...store }))
    };
}
export function all(store) {
    return derived(store, ({ ids, entities }) => {
        const results = [];
        for (let i = 0; i < ids.length; i++) {
            results.push(entities[ids[i]]);
        }
        return results;
    });
}
export function where(store, filterFn) {
    return derived(store, ({ ids, entities }) => {
        const results = [];
        for (let i = 0; i < ids.length; i++) {
            if (filterFn(entities[ids[i]])) {
                results.push(entities[ids[i]]);
            }
        }
        return results;
    });
}
export function createDataStore() {
    const taskStore = createEntityStore();
    const rowStore = createEntityStore();
    const timeRangeStore = createEntityStore();
    const allTasks = all(taskStore);
    const allRows = all(rowStore);
    const allTimeRanges = all(timeRangeStore);
    const rowTaskCache = derived(allTasks, $allTasks => {
        const cache = {};
        for (let i = 0; i < $allTasks.length; i++) {
            const task = $allTasks[i];
            if (!cache[task.model.resourceId]) {
                cache[task.model.resourceId] = [];
            }
            cache[task.model.resourceId].push(task.model.id);
        }
        return cache;
    });
    return {
        taskStore,
        rowStore,
        timeRangeStore,
        allTasks,
        allRows,
        allTimeRanges,
        rowTaskCache
    };
}
