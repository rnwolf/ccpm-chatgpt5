export interface DraggableOptions {
    container: HTMLElement;
    dragAllowed: MaybeAccessor<boolean>;
    resizeAllowed: MaybeAccessor<boolean>;
    resizeHandleWidth?: number;
    getX?: (event?: MouseEvent) => number;
    getY?: (event?: MouseEvent) => number;
    getWidth?: () => number;
    onDown?(event?: DownDropEvent): void;
    onResize?(event?: ResizeEvent): void;
    onDrag?(event?: DragEvent): void;
    onMouseUp?(): void;
    onDrop?(event?: DownDropEvent): void;
}
export interface DownDropEvent {
    mouseEvent: MouseEvent;
    x: number;
    y: number;
    width: number;
    resizing: boolean;
    dragging: boolean;
}
export interface DragEvent {
    x: number;
    y: number;
    event?: MouseEvent;
}
export interface ResizeEvent {
    x: number;
    width: number;
    event?: MouseEvent;
}
export type Direction = 'left' | 'right' | undefined;
type MaybeAccessor<T> = T | (() => T);
/**
 * Applies dragging interaction to gantt elements
 */
export declare function useDraggable(node: HTMLElement, options: DraggableOptions): {
    destroy(): void;
};
/**
 * Applies dragging interaction to gantt elements
 */
export declare function createDraggable(options: DraggableOptions): (event: PointerEvent) => void;
export {};
