type Options = {
    enabled?: boolean;
    container: HTMLElement;
    boundsContainer?: HTMLElement;
    onMove(e: MoveEvent): void;
    onEnd(e: MoveEvent): void;
};
export type MoveEvent = {
    from: number;
    to: number;
    x: number;
    y: number;
    width: number;
};
export declare function useCreateTask(): (node: HTMLElement, options: Options) => {
    destroy(): void;
    update(opts: Options): void;
};
export {};
