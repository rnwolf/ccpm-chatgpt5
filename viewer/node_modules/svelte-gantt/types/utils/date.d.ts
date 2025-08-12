export interface SvelteGanttDateAdapter {
    roundTo(date: number, unit: string, offset: number): number;
    format(date: number, format: string): string;
}
export declare function startOf(date: number, unit: string): number;
export declare function getDuration(unit: string, offset?: number): number;
export declare function getDurationV2(unit: string, offset?: number, date?: any): number;
/**
 *
 * @param from Interval start
 * @param to Interval end
 * @param unit Column unit
 * @param offset Column spacing
 * @param highlightedDurations
 * @returns
 */
export declare function getAllPeriods(from: number, to: number, unit: string, offset?: number, highlightedDurations?: any): {
    from: number;
    to: number;
    duration: number;
    isHighlighted: boolean;
}[];
