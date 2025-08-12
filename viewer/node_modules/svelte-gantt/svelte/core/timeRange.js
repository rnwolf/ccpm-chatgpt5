export class TimeRangeFactory {
    columnService;
    constructor(columnService) {
        this.columnService = columnService;
    }
    create(model) {
        // enable dragging
        model.resizable = model.resizable === undefined ? true : model.resizable;
        const left = this.columnService.getPositionByDate(model.from);
        const right = this.columnService.getPositionByDate(model.to);
        return {
            model,
            left: left,
            width: right - left,
            resizing: false
        };
    }
}
