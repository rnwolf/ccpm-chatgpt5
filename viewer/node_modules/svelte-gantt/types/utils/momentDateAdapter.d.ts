import { SvelteGanttDateAdapter } from './date';
/**
 * Date adapter that uses MomentJS
 */
export declare class MomentSvelteGanttDateAdapter implements SvelteGanttDateAdapter {
    moment: any;
    constructor(moment: any);
    format(date: number, format: string): string;
    roundTo(date: number, unit: string, offset: number): number;
}
