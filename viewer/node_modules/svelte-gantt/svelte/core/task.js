export function createTaskFactory(params) {
    return {
        createTask: (model) => createTask(model, params),
        reflectTask: (task, targetRow) => reflectTask(task, targetRow, params),
    };
}
export function createTask(model, params) {
    model.amountDone = model.amountDone ?? 0;
    model.showButton = model.showButton ?? false;
    model.buttonClasses = model.buttonClasses ?? '';
    model.buttonHtml = model.buttonHtml ?? '';
    const left = params.getPositionByDate(model.from) | 0;
    const right = params.getPositionByDate(model.to) | 0;
    const row = params.rowEntities[model.resourceId];
    const height = (row ? row.height : undefined) - 2 * params.rowPadding;
    const top = (row ? row.y : -1000) + params.rowPadding;
    return {
        model,
        left: left,
        width: right - left,
        height,
        top,
    };
}
export function overlap(left, right) {
    return !(left.left + left.width <= right.left || left.left >= right.left + right.width);
}
export function reflectTask(task, targetRow, params) {
    const reflectedId = `reflected-task-${String(task.model.id)}-${String(targetRow.model.id)}`;
    const model = {
        ...task.model,
        resourceId: targetRow.model.id,
        id: reflectedId,
        enableDragging: false
    };
    return {
        ...task,
        model,
        top: targetRow.y + params.rowPadding,
        reflected: true,
        reflectedOnParent: false,
        reflectedOnChild: true,
        originalId: task.model.id
    };
}
