// dateAdapter.js
import moment from 'moment';

export class MomentSvelteGanttDateAdapter {
  format(date, formatStr) {
    return moment(date).format(formatStr);
  }
  roundTo(date, unit) {
    return moment(date).startOf(unit).toDate();
  }
  add(date, amount, unit) {
    return moment(date).add(amount, unit).toDate();
  }
  diff(date1, date2, unit) {
    return moment(date1).diff(moment(date2), unit);
  }
  now() {
    return moment().toDate();
  }
}
