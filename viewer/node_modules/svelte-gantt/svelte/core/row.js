export function createRows(rows, params) {
    const context = { y: 0, result: [] };
    createChildRows(rows, context, params);
    return context.result;
}
function createChildRows(rowModels, context, params, parent = null, level = 0, parents = []) {
    const rowsAtLevel = [];
    const allRows = [];
    if (parent) {
        parents = [...parents, parent];
    }
    for (const model of rowModels) {
        const row = createRow(model, context.y, params);
        context.result.push(row);
        rowsAtLevel.push(row);
        allRows.push(row);
        row.childLevel = level;
        row.parent = parent;
        row.allParents = parents;
        if (parent) {
            // when row is hidden, other rows (y-pos) move upward
            row.hidden = !(parent.model.expanded || parent.model.expanded == null) || parent.hidden != null && parent.hidden;
        }
        if (!row.hidden) {
            context.y += row.height;
        }
        if (model.children) {
            const nextLevel = createChildRows(model.children, context, params, row, level + 1, parents);
            row.children = nextLevel.rows;
            row.allChildren = nextLevel.allRows;
            allRows.push(...nextLevel.allRows);
        }
    }
    return {
        rows: rowsAtLevel,
        allRows
    };
}
function createRow(model, y, params) {
    // defaults
    // height of row element
    const height = model.height ?? params.rowHeight;
    return {
        model: model,
        y,
        height,
    };
}
export function expandRow(row) {
    row.model.expanded = true;
    if (row.children)
        show(row.children);
}
export function collapseRow(row) {
    row.model.expanded = false;
    if (row.children)
        hide(row.children);
}
function hide(children) {
    for (const row of children) {
        if (row.children)
            hide(row.children);
        row.hidden = true;
    }
}
function show(children, hidden = false) {
    for (const row of children) {
        if (row.children)
            show(row.children, !row.model.expanded);
        row.hidden = hidden;
    }
}
