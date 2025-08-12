/// <reference types="svelte" />
import { ContextMenu } from '../ui';
export declare class ContextMenuManager {
    current: ContextMenu;
    constructor();
    open(actions: any, position: any): ContextMenu<any, any, any>;
    close(): void;
}
