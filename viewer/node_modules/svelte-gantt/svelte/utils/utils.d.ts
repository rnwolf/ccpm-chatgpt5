import { SvelteGanttDateAdapter } from './date';
type UtilsParams = Readonly<{
    from: number;
    to: number;
    width: number;
    magnetOffset: number;
    magnetUnit: string;
    magnetDuration: number;
    dateAdapter: SvelteGanttDateAdapter;
}>;
export declare function createUtils(params: UtilsParams): {
    /**
     * Returns position of date on a line if from and to represent length of width
     * @param {*} date
     */
    getPositionByDate(date: number): number;
    getDateByPosition(x: any): number;
    roundTo(date: number): number;
};
export type GanttUtils = ReturnType<typeof createUtils>;
export declare function getPositionByDate(date: number, from: number, to: number, width: number): number;
export declare function getDateByPosition(x: number, from: number, to: number, width: number): number;
export declare function getIndicesOnly<T, C = number | Date>(input: T[], value: C, comparer: {
    (T: T): C;
}, strict?: boolean): number[];
export declare function get<T, C = number | Date>(input: T[], value: C, comparer: {
    (T: T): C;
}, strict?: boolean): T[];
export declare function isDraggable(item: {
    draggable?: boolean;
    enableDragging?: boolean;
}): boolean;
export declare function isResizable(item: {
    resizable?: boolean;
    enableResize?: boolean;
}): boolean;
export {};
