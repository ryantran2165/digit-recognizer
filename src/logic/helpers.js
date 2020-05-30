import Matrix from "./matrix";

/**
 * Returns the sigmoid of the given number.
 * @param {Number} x The number to apply the sigmoid function to
 * @return {Number} the sigmoid of the given number
 */
export const sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x));
};

/**
 * Returns the sigmoid derivative of the given number.
 * @param {Number} x The number to apply the sigmoid derivative function to
 * @return {Number} the sigmoid derivative of the given number
 */
export const sigmoidDerivative = (x) => {
  return sigmoid(x) * (1 - sigmoid(x));
};

/**
 * The quadratic cost function (unused).
 */
export class QuadraticCost {
  /**
   * Returns the cost.
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @return the cost
   */
  static fn = (a, y) => {
    const diffMatrix = Matrix.sub(a, y);
    let cost = 0;
    for (let r = 0; r < diffMatrix.rows; r++) {
      for (let c = 0; c < diffMatrix.cols; c++) {
        cost += 0.5 * diffMatrix.data[r][c] ** 2;
      }
    }
    return cost;
  };

  /**
   * Returns the error of the output layer.
   * @param {Matrix} z The z matrix of the output layer
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @return the error of the output layer
   */
  static delta = (z, a, y) => {
    return Matrix.sub(a, y).mul(Matrix.map(z, sigmoidDerivative));
  };
}

/**
 * The cross entropy cost function (currently used).
 */
export class CrossEntropyCost {
  static fn = (a, y) => {
    /**
     * Returns the cost.
     * @param {Matrix} a The activation of the output layer
     * @param {Matrix} y The desired output
     * @return the cost
     */
    const yIsOne = Matrix.mul(
      Matrix.transpose(y),
      Matrix.map(a, (x) => Math.log(x))
    );
    const yIsZero = Matrix.mul(
      Matrix.transpose(Matrix.map(y, (x) => 1 - x)),
      Matrix.map(a, (x) => Math.log(1 - x))
    );
    return -(yIsOne.data[0][0] + yIsZero.data[0][0]);
  };

  /**
   * Returns the error of the output layer.
   * @param {Matrix} z The z matrix of the output layer
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @return the error of the output layer
   */
  static delta = (z, a, y) => {
    return Matrix.sub(a, y);
  };
}

/**
 * Fisher-Yates shuffle.
 * @param {Array} arr The array to shuffle
 * @return {Array} The shuffled array
 */
export const shuffle = (arr) => {
  for (let i = arr.length - 1; i > 0; i--) {
    const randomIndex = Math.floor(Math.random() * (i + 1));
    const temp = arr[i];
    arr[i] = arr[randomIndex];
    arr[randomIndex] = temp;
  }
  return arr;
};
