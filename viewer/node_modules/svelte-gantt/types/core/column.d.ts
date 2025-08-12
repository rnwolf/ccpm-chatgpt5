export interface HighlightedDurations {
    unit: string;
    fractions: number[];
}
export interface Column {
    from: number;
    to: number;
    left: number;
    width: number;
    bgHighlightColor?: boolean;
    /**
     * Duration in milliseconds
     */
    duration: number;
}
type ColumnServiceParams = {
    readonly columns: Column[];
    readonly magnetDuration: number;
};
export declare function createColumnService(params: ColumnServiceParams): {
    getColumnByDate: (date: number) => Column;
    getColumnByPosition: (x: number) => Column;
    getPositionByDate(date: number): number;
    getDateByPosition(x: number): number;
    /**
     * TODO: remove, currently unused
     * @param {number} date - Date
     * @returns {number} rounded date passed as parameter
     */
    roundTo(date: number): number;
};
export declare function findByPosition(columns: Column[], x: number): Column[];
export declare function findByDate(columns: Column[], x: number): Column[];
export type ColumnService = ReturnType<typeof createColumnService>;
export {};
