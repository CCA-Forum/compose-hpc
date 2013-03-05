int get(int *arr, int w, int x, int y) {
  return arr[y * w + x];
}

void put(int *arr, int w, int x, int y, int v) {
  arr[y * w + x] = v;
}
