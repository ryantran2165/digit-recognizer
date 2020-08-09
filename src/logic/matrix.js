class Matrix {
  /**
   * Creates an empty matrix of all zeros or from the given Matrix.
   * @param {number} rows The number of rows
   * @param {number} cols The number of columns
   * @param {Matrix} matrix A Matrix to copy
   */
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

  /**
   * Element-wise addition by a Matrix or scaler.
   * @param {(Matrix | number)} matrix The Matrix or scaler to add by
   * @return {Matrix} The Matrix for chaining
   */
  add(matrix) {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Add: matrix dimensions must match.");
        return;
      }
      return this.map((x, i, j) => x + matrix.data[i][j]);
    }
    return this.map((x) => x + matrix);
  }

  /**
   * Element-wise addition into a new Matrix.
   * @param {Matrix} matrix1 The first Matrix
   * @param {Matrix} matrix2 The second Matrix
   * @return {Matrix} A new Matrix from the addition of the two given matrices
   */
  static add(matrix1, matrix2) {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      console.log("Add: matrix dimensions must match.");
      return;
    }
    return new Matrix(matrix1.rows, matrix1.cols).map(
      (_, i, j) => matrix1.data[i][j] + matrix2.data[i][j]
    );
  }

  /**
   * Element-wise subtraction by a Matrix or scaler.
   * @param {(Matrix | number)} matrix The Matrix or scaler to subtract by
   * @return {Matrix} The Matrix for chaining
   */
  sub(matrix) {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Subtract: matrix dimensions must match.");
        return;
      }
      // Matrix
      return this.map((x, i, j) => x - matrix.data[i][j]);
    }
    //Scaler
    return this.map((x) => x - matrix);
  }

  /**
   * Element-wise subtraction into a new Matrix.
   * @param {Matrix} matrix1 The first Matrix
   * @param {Matrix} matrix2 The second Matrix
   * @return {Matrix} A new Matrix from the subtraction of the two given matrices
   */
  static sub(matrix1, matrix2) {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      console.log("Subtract: matrix dimensions must match.");
      return;
    }
    return new Matrix(matrix1.rows, matrix1.cols).map(
      (_, i, j) => matrix1.data[i][j] - matrix2.data[i][j]
    );
  }

  /**
   * Element-wise multiplication by a Matrix or scaler.
   * @param {(Matrix | number)} matrix The scaler or Matrix to multiply by
   * @return {Matrix} The Matrix for chaining
   */
  mul(matrix) {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Multiply: matrix dimensions must match.");
        return;
      }
      // Matrix
      return this.map((x, i, j) => x * matrix.data[i][j]);
    }
    // Scaler
    return this.map((x) => x * matrix);
  }

  /**
   * Matrix multiplication into a new Matrix.
   * @param {Matrix} matrix1 The first Matrix
   * @param {Matrix} matrix2 The second Matrix
   * @return {Matrix} A new Matrix from the matrix multiplication of the two given matrices
   */
  static mul(matrix1, matrix2) {
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
  }

  /**
   * Element-wise division by a Matrix or scaler.
   * @param {(Matrix | number)} matrix The Matrix or scaler to divide by
   * @return {Matrix} The Matrix for chaining
   */
  div(matrix) {
    if (matrix instanceof Matrix) {
      if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
        console.log("Division: matrix dimensions must match.");
        return;
      }
      // Matrix
      return this.map((x, i, j) => x / matrix.data[i][j]);
    }
    // Scaler
    return this.map((x) => x / matrix);
  }

  /**
   * Element-wise division into a new Matrix.
   * @param {Matrix} matrix1 The first Matrix
   * @param {Matrix} matrix2 The second Matrix
   * @return {Matrix} A new Matrix from the division of the two given matrices
   */
  static div(matrix1, matrix2) {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      console.log("Division: matrix dimensions must match.");
      return;
    }
    return new Matrix(matrix1.rows, matrix1.cols).map(
      (_, i, j) => matrix1.data[i][j] / matrix2.data[i][j]
    );
  }

  /**
   * Transposes the Matrix.
   * @param {Matrix} matrix The Matrix to transpose
   * @return {Matrix} The Matrix for chaining
   */
  static transpose(matrix) {
    return new Matrix(matrix.cols, matrix.rows).map(
      (_, i, j) => matrix.data[j][i]
    );
  }

  /**
   * Converts the array into a column vector.
   * @param {Array} arr The array to convert to a vector
   * @return {Matrix} The array as a column vector
   */
  static vectorFromArray(arr) {
    return new Matrix(arr.length, 1).map((_, i) => arr[i]);
  }

  /**
   * Converts the array into a Matrix of the given dimensions.
   * @param {Array} arr The array to convert to a Matrix
   * @param {number} rows The number of rows for the new Matrix
   * @param {number} cols The number of columns for the new Matrix
   * @return {Matrix} The array as a Matrix of the given dimension
   */
  static matrixFromArray(arr, rows, cols) {
    return new Matrix(rows, cols).map((_, i, j) => arr[i * cols + j]);
  }

  /**
   * Converts this Matrix to an array.
   * @return {Array} This Matrix as an array
   */
  toArray() {
    const arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  /**
   * Returns a copy of this Matrix.
   * @return {Matrix} A copy of this Matrix
   */
  copy() {
    return new Matrix(this.rows, this.cols).map((_, i, j) => this.data[i][j]);
  }

  /**
   * Applies the function to the Matrix.
   * @param {Function} func The function to apply to the Matrix
   * @return {Matrix} The Matrix for chaining
   */
  map(func) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = func(this.data[i][j], i, j);
      }
    }
    return this;
  }

  /**
   * Applies the function to the Matrix and returns a new Matrix.
   * @param {Matrix} matrix The Matrix to apply the function to
   * @param {Function} func The function to apply to the Matrix
   * @return {Matrix} A new Matrix from applying the function to the Matrix
   */
  static map(matrix, func) {
    return new Matrix(matrix.rows, matrix.cols).map((_, i, j) =>
      func(matrix.data[i][j], i, j)
    );
  }

  /**
   * Randomize [-1, 1].
   * @return {Matrix} The Matrix for chaining
   */
  randomize() {
    return this.map(() => Math.random() * 2 - 1);
  }

  /**
   * Box-Muller Transform for normal distribution, mean = 0, variance = 1.
   * @return {Matrix} The Matrix for chaining
   */
  randomizeNormal() {
    return this.map(() => {
      let u = 0;
      let v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    });
  }

  /**
   * Prints the Matrix as a table.
   * @return {Matrix} The Matrix for chaining
   */
  print() {
    console.table(this.data);
    return this;
  }

  /**
   * Returns a region of this Matrix.
   * @param {number} x The column offset
   * @param {number} y The row offset
   * @param {number} w The width of the region to extract
   * @param {number} h The height of the region to extract
   * @return {Matrix} The region of this Matrix
   */
  getRegion(x, y, w, h) {
    const region = new Matrix(h, w);
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        region.data[r][c] = this.data[y + r][x + c];
      }
    }
    return region;
  }

  /**
   * Returns the max value and the corresponding row and column.
   * @return {Array} The max value and the corresponding row and column.
   */
  max() {
    let max = Number.NEGATIVE_INFINITY;
    let maxR = -1;
    let maxC = -1;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (this.data[r][c] > max) {
          max = this.data[r][c];
          maxR = r;
          maxC = c;
        }
      }
    }
    return [max, maxR, maxC];
  }
}

export default Matrix;
