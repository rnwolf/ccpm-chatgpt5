export function createUtils(params) {
    return {
        /**
         * Returns position of date on a line if from and to represent length of width
         * @param {*} date
         */
        getPositionByDate(date) {
            return getPositionByDate(date, params.from, params.to, params.width);
        },
        getDateByPosition(x) {
            return getDateByPosition(x, params.from, params.to, params.width);
        },
        roundTo(date) {
            if (params.dateAdapter) {
                return params.dateAdapter.roundTo(date, params.magnetUnit, params.magnetOffset);
            }
            // this does not consider the timezone, rounds only to the UTC time
            // let value = Math.round((date - 7200000) / params.magnetDuration) * params.magnetDuration;
            // cases where rounding to day or timezone offset is not rounded, this won't work
            return null;
        }
    };
}
export function getPositionByDate(date, from, to, width) {
    if (!date) {
        return undefined;
    }
    const durationTo = date - from;
    const durationToEnd = to - from;
    return (durationTo / durationToEnd) * width;
}
export function getDateByPosition(x, from, to, width) {
    const durationTo = (x / width) * (to - from);
    const dateAtPosition = from + durationTo;
    return dateAtPosition;
}
// Returns the object on the left and right in an array using the given cmp function.
// The compare function defined which property of the value to compare (e.g.: c => c.left)
export function getIndicesOnly(input, value, comparer, strict) {
    let lo = -1;
    let hi = input.length;
    while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (strict ? comparer(input[mid]) < value : comparer(input[mid]) <= value) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }
    if (!strict && input[lo] !== undefined && comparer(input[lo]) === value) {
        hi = lo;
    }
    return [lo, hi];
}
export function get(input, value, comparer, strict) {
    const res = getIndicesOnly(input, value, comparer, strict);
    return [input[res[0]], input[res[1]]];
}
export function isDraggable(item) {
    return item.draggable ?? item.enableDragging ?? true;
}
export function isResizable(item) {
    return item.resizable ?? item.enableResize ?? true;
}
