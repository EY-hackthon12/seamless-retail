// runs once when the module is loaded (outside React)
export const precomputedTiles: boolean[] = (() => {
  const arr: boolean[] = [];
  for (let i = 0; i < 400; i++) {
    // eslint-disable-next-line no-restricted-globals
    arr.push(Math.random() > 0.85);
  }
  return arr;
})();
