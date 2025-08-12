import { get } from '../utils/utils';
export function createColumnService(params) {
    function getColumnByDate(date) {
        const pair = findByDate(params.columns, date);
        return !pair[0] ? pair[1] : pair[0];
    }
    function getColumnByPosition(x) {
        const pair = findByPosition(params.columns, x);
        return !pair[0] ? pair[1] : pair[0];
    }
    return {
        getColumnByDate,
        getColumnByPosition,
        getPositionByDate(date) {
            if (!date)
                return null;
            const column = getColumnByDate(date);
            let durationTo = date - column.from;
            const position = (durationTo / column.duration) * column.width;
            //multiples - skip every nth col, use other duration
            return column.left + position;
        },
        getDateByPosition(x) {
            const column = getColumnByPosition(x);
            x = x - column.left;
            let positionDuration = (column.duration / column.width) * x;
            const date = column.from + positionDuration;
            return date;
        },
        /**
         * TODO: remove, currently unused
         * @param {number} date - Date
         * @returns {number} rounded date passed as parameter
         */
        roundTo(date) {
            let value = Math.round(date / params.magnetDuration) * params.magnetDuration;
            return value;
        }
    };
}
export function findByPosition(columns, x) {
    const result = get(columns, x, c => c.left);
    return result;
}
export function findByDate(columns, x) {
    const result = get(columns, x, c => c.from);
    return result;
}
