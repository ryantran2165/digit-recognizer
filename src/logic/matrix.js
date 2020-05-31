class Matrix {
  constructor(rows, cols, matrix) {
    if (matrix) {
      // Deep copy
      this.rows = matrix.rows;
      this.cols = matrix.cols;
      this.data = [];
      for (let i = 0; i < this.rows; i++) {
        this.data.push([]);
        for (let j = 0; j < this.cols; j++) {
          this.data[i].push(matrix.data[i][j]);
        }
      }
    } else {
      this.rows = rows;
      this.cols = cols;
      this.data = Array(rows)
        .fill()
        .map(() => Array(cols).fill(0));
    }
  }

  add = (matrix) => {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Add: matrix dimensions must match.");
        return;
      }
      return this.map((x, i, j) => x + matrix.data[i][j]);
    }
    return this.map((x) => x + matrix);
  };

  static add = (matrix1, matrix2) => {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      console.log("Add: matrix dimensions must match.");
      return;
    }
    return new Matrix(matrix1.rows, matrix1.cols).map(
      (_, i, j) => matrix1.data[i][j] + matrix2.data[i][j]
    );
  };

  sub = (matrix) => {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Subtract: matrix dimensions must match.");
        return;
      }
      return this.map((x, i, j) => x - matrix.data[i][j]);
    }
    return this.map((x) => x - matrix);
  };

  static sub = (matrix1, matrix2) => {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      console.log("Subtract: matrix dimensions must match.");
      return;
    }
    return new Matrix(matrix1.rows, matrix1.cols).map(
      (_, i, j) => matrix1.data[i][j] - matrix2.data[i][j]
    );
  };

  mul = (matrix) => {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Multiply: matrix dimensions must match.");
        return;
      }
      return this.map((x, i, j) => x * matrix.data[i][j]);
    }
    return this.map((x) => x * matrix);
  };

  static mul = (matrix1, matrix2) => {
    if (matrix1.cols !== matrix2.rows) {
      console.log(
        "Multiply: first matrix's columns must match second matrix's rows"
      );
      return;
    }
    return new Matrix(matrix1.rows, matrix2.cols).map((_, i, j) => {
      let sum = 0;
      for (let k = 0; k < matrix1.cols; k++) {
        sum += matrix1.data[i][k] * matrix2.data[k][j];
      }
      return sum;
    });
  };

  div = (matrix) => {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Division: matrix dimensions must match.");
        return;
      }
      return this.map((x, i, j) => x / matrix.data[i][j]);
    }
    return this.map((x) => x / matrix);
  };

  static transpose = (matrix) => {
    return new Matrix(matrix.cols, matrix.rows).map(
      (_, i, j) => matrix.data[j][i]
    );
  };

  static vectorFromArray = (arr) => {
    return new Matrix(arr.length, 1).map((_, i) => arr[i]);
  };

  toArray = () => {
    const arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  };

  copy = () => {
    return new Matrix(this.rows, this.cols).map((_, i, j) => this.data[i][j]);
  };

  map = (func) => {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = func(this.data[i][j], i, j);
      }
    }
    return this;
  };

  static map = (matrix, func) => {
    return new Matrix(matrix.rows, matrix.cols).map((_, i, j) =>
      func(matrix.data[i][j], i, j)
    );
  };

  randomize = () => {
    return this.map(() => Math.random() * 2 - 1);
  };

  // Box-Muller Transform for normal distribution, mean = 0, variance = 1
  randomizeNormal = () => {
    return this.map(() => {
      let u = 0;
      let v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    });
  };

  print = () => {
    console.table(this.data);
    return this;
  };
}

export default Matrix;
