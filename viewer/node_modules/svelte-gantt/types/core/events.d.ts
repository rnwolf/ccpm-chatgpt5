type DelegatedCallback = (e: MouseEvent, data: string, target: Element) => void;
export declare function createDelegatedEventDispatcher(): {
    onDelegatedEvent(type: any, attr: any, callback: DelegatedCallback): void;
    offDelegatedEvent(type: any, attr: any): void;
    onEvent(e: MouseEvent): void;
};
export {};
