import { isDraggable } from '../../utils/utils';
import { useDraggable } from '../../core/drag';
import { getRelativePos, getRowAtPoint } from '../../utils/dom';
const defaults = {
    enabled: true,
    elementContent: () => {
        const element = document.createElement('div');
        element.innerHTML = 'New Task';
        Object.assign(element.style, {
            position: 'absolute',
            background: '#eee',
            padding: '0.5em 1em',
            fontSize: '12px',
            pointerEvents: 'none'
        });
        return element;
    }
};
export class SvelteGanttExternal {
    draggable;
    element;
    options;
    constructor(node, options) {
        this.options = Object.assign({}, defaults, options);
        this.draggable = useDraggable(node, {
            onDrag: this.onDrag.bind(this),
            dragAllowed: () => this.options.enabled,
            resizeAllowed: false,
            onDrop: this.onDrop.bind(this),
            container: document.body,
            getX: (event) => event.pageX,
            getY: (event) => event.pageY,
            getWidth: () => 0
        });
    }
    onDrag({ x, y }) {
        if (!this.element) {
            this.element = this.options.elementContent();
            document.body.appendChild(this.element);
            this.options.dragging = true;
        }
        this.element.style.top = y + 'px';
        this.element.style.left = x + 'px';
    }
    onDrop(event) {
        const gantt = this.options.gantt;
        const rowId = getRowAtPoint(event.mouseEvent);
        const targetRow = gantt.getRow(rowId);
        if (targetRow && isDraggable(targetRow.model)) {
            const mousePos = getRelativePos(gantt.getRowContainer(), event.mouseEvent);
            const date = gantt.utils.getDateByPosition(mousePos.x);
            this.options.onsuccess?.(targetRow, date, gantt);
        }
        else {
            this.options.onfail?.();
        }
        document.body.removeChild(this.element);
        this.options.dragging = false;
        this.element = null;
    }
}
