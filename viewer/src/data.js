export const rows = [
  { id: 1, label: "Project A" },
  { id: 2, label: "Project B" }
];

export const tasks = [
  {
    id: 1,
    label: "Design",
    from: new Date(2024, 5, 11),
    to: new Date(2024, 5, 14),
    progress: 50,
    rowId: 1
  },
  {
    id: 2,
    label: "Development",
    from: new Date(2024, 5, 15),
    to: new Date(2024, 5, 25),
    progress: 20,
    rowId: 2
  }
];

export const timeRanges = [
  {
    id: 'range1',
    label: 'Sprint 1',
    from: new Date(2024, 5, 12),
    to: new Date(2024, 5, 13),
    rowId: 1
  }
];
