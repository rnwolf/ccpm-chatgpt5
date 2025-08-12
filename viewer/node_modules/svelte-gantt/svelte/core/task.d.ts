import type { SvelteRow } from './row';
export interface TaskModel {
    /** id of task, every task needs to have a unique one */
    id: PropertyKey;
    resourceId: PropertyKey;
    /** date task starts on */
    from: number;
    /** date task ends on */
    to: number;
    /**
     * completion %, indicated on task
     * @deprecated
     */
    amountDone?: number;
    /** css classes */
    classes?: string | string[];
    /** label of task */
    label?: string;
    /** html content of task, will override label */
    html?: string;
    /**
     * show button bar
     * @deprecated
     **/
    showButton?: boolean;
    /**
     * button classes, useful for fontawesome icons
     * @deprecated
     **/
    buttonClasses?: string | string[];
    /**
     * html content of button
     * @deprecated
     */
    buttonHtml?: string;
    /**
     * enable dragging of task
     * @deprecated use draggable
     **/
    enableDragging?: boolean;
    /**
     * enable dragging of task
     **/
    draggable?: boolean;
    /**
     * enable resizing of task
     * @deprecated use resizable
     */
    enableResize?: boolean;
    /**
     * enable resizing of task
     */
    resizable?: boolean;
    /**
     * label displayed below
     * @deprecated
     **/
    labelBottom?: string;
    type?: 'milestone' | 'task';
    stickyLabel?: boolean;
}
export interface SvelteTask {
    model: TaskModel;
    left: number;
    top: number;
    width: number;
    height: number;
    reflected?: boolean;
    reflectedOnParent?: boolean;
    reflectedOnChild?: boolean;
    originalId?: PropertyKey;
}
type CreateTaskParams = {
    rowPadding: number;
    rowEntities: {
        [rowId: PropertyKey]: SvelteRow;
    };
    getPositionByDate(date: any): number;
};
export declare function createTaskFactory(params: CreateTaskParams): {
    createTask: (model: TaskModel) => SvelteTask;
    reflectTask: (task: SvelteTask, targetRow: SvelteRow) => SvelteTask;
};
export declare function createTask(model: TaskModel, params: CreateTaskParams): SvelteTask;
export declare function overlap(left: SvelteTask, right: SvelteTask): boolean;
export declare function reflectTask(task: SvelteTask, targetRow: SvelteRow, params: CreateTaskParams): SvelteTask;
export {};
