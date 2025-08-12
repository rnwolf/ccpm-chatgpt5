import { SvelteGanttDateAdapter } from './date';
export declare class DefaultSvelteGanttDateAdapter implements SvelteGanttDateAdapter {
    format(date: number, format: string): string;
    /**
     * Rounds the date down to the nearest unit
     *
     * Note: This does not consider the timezone, rounds only to the UTC time, which makes it incorrect to round to day start or half hour time zones
     */
    roundTo(date: number, unit: string, offset: number): number;
}
