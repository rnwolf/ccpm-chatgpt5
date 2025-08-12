import { writable } from 'svelte/store';
export class SelectionManager {
    taskStore;
    _selectedTasks = writable({});
    constructor(taskStore) {
        this.taskStore = taskStore;
    }
    selectSingle(taskId) {
        this.unSelectTasks();
        this._selectedTasks.set({ [taskId]: true });
    }
    toggleSelection(taskId) {
        this._selectedTasks.update(selections => ({
            ...selections,
            [taskId]: !selections[taskId]
        }));
    }
    unSelectTasks() {
        this._selectedTasks.set({});
    }
}
