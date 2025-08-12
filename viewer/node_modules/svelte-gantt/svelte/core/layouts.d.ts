import type { SvelteRow } from './row';
import type { EntityState } from './store';
import type { SvelteTask } from './task';
export type LayoutParams = {
    taskStore: EntityState<SvelteTask>;
    rowStore: EntityState<SvelteRow>;
    rowTasks: {
        [rowId: PropertyKey]: PropertyKey[];
    };
    rowHeight: number;
    rowPadding: number;
    rowReflectedTasks: {
        [rowId: PropertyKey]: SvelteTask[];
    };
    invalidatedTasks: {
        [rowId: PropertyKey]: boolean;
    };
    invalidatedRows: {
        [rowId: PropertyKey]: boolean;
    };
    invalidateFull?: boolean;
};
type LayoutResult = {
    changed: boolean;
};
/**
 * Layouts tasks - overlapping tasks display one over another
 * @param params
 */
export declare function overlap(params: LayoutParams): LayoutResult;
/**
 * Layouts tasks - overlapping tasks display in the same row, but shrink to not overlap with eachother
 * @param params
 */
export declare function pack(params: LayoutParams): void;
/**
 * Layouts tasks - overlapping tasks display in the same row, but row is expanded to fit them
 * @param params
 * @returns
 */
export declare function expand(params: LayoutParams): void;
export {};
