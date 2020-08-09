import Matrix from "../matrix";

/**
 * The sigmoid activation function and derivative.
 */
export class SigmoidActivation {
  /**
   * Returns the matrix mapped with sigmoid.
   * @param {Matrix} z The matrix to apply the sigmoid function to
   * @return {Matrix} The matrix mapped with sigmoid
   */
  static fn(z) {
    return Matrix.map(z, (z_i) => 1 / (1 + Math.exp(-z_i)));
  }

  /**
   * Returns the matrix mapped with sigmoid derivative.
   * @param {Matrix} z The matrix to apply the sigmoid derivative function to
   * @return {Matrix} The matrix mapped with sigmoid derivative
   */
  static derivative(z) {
    const sigmoid = SigmoidActivation.fn(z);
    return sigmoid.mul(Matrix.map(sigmoid, (a_i) => 1 - a_i));
  }
}

/**
 * The ReLU activation function and derivative.
 */
export class ReLUActivation {
  /**
   * Returns the matrix mapped with ReLU.
   * @param {Matrix} z The matrix to apply the ReLu function to
   * @return {Matrix} The matrix mapped with ReLu
   */
  static fn(z) {
    return Matrix.map(z, (z_i) => Math.max(0, z_i));
  }

  /**
   * Returns the matrix mapped with ReLU derivative.
   * @param {Matrix} z The matrix to apply the ReLU derivative function to
   * @return {Matrix} The matrix mapped with ReLU derivative
   */
  static derivative(z) {
    return Matrix.map(z, (z_i) => (z_i > 0 ? 1 : 0));
  }
}

/**
 * The ReLU activation function and derivative.
 */
export class LeakyReLUActivation {
  /**
   * Returns the matrix mapped with leaky ReLU.
   * @param {Matrix} z The matrix to apply the leaky ReLu function to
   * @return {Matrix} The matrix mapped with leaky ReLu
   */
  static fn(z) {
    return Matrix.map(z, (z_i) => (z_i > 0 ? z_i : 0.01 * z_i));
  }

  /**
   * Returns the matrix mapped with leaky ReLU derivative.
   * @param {Matrix} z The matrix to apply the leaky ReLU derivative function to
   * @return {Matrix} The matrix mapped with leaky ReLU derivative
   */
  static derivative(z) {
    return Matrix.map(z, (z_i) => (z_i > 0 ? 1 : 0.01));
  }
}

/**
 * The softmax activation function and derivative.
 */
export class SoftmaxActivation {
  /**
   * Returns the matrix mapped with softmax.
   * @param {Matrix} z The matrix to apply the softmax function to
   * @return {Matrix} The matrix mapped with softmax
   */
  static fn(z) {
    let sum = 0;
    for (let r = 0; r < z.rows; r++) {
      sum += Math.exp(z.data[r][0]);
    }
    return Matrix.map(z, (z_i) => Math.exp(z_i) / sum);
  }

  /**
   * Returns the softmax derivative of the given number.
   * @param {Matrix} z The matrix to apply the softmax derivative function to
   * @return {Matrix} The matrix mapped with softmax derivative
   */
  static derivative(z) {
    const softmax = SoftmaxActivation.fn(z);
    return softmax.mul(Matrix.map(softmax, (a_i) => 1 - a_i));
  }
}

/**
 * The quadratic loss function and output layer error.
 */
export class QuadraticLoss {
  /**
   * Returns the loss = sum over all (0.5 * (a - y)^2).
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @return {number} The loss
   */
  static fn(a, y) {
    const diffMatrix = Matrix.sub(a, y);
    let loss = 0;
    for (let r = 0; r < diffMatrix.rows; r++) {
      for (let c = 0; c < diffMatrix.cols; c++) {
        loss += 0.5 * diffMatrix.data[r][c] ** 2;
      }
    }
    return loss;
  }

  /**
   * Returns the output error = (a - y) hadamardProduct outputActivationFunctionDerivative(z).
   * @param {Matrix} z The z of the output layer
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @param {Function} outputActivationFunction The activation function of the output layer
   * @return {Matrix} The output error
   */
  static outputError(z, a, y, outputActivationFunction) {
    const lossDerivativeWRTa = Matrix.sub(a, y);
    return lossDerivativeWRTa.mul(outputActivationFunction.derivative(z));
  }
}

const EPSILON = 10 ** -100;

/**
 * The binary cross entropy loss function and output layer error.
 */
export class BinaryCrossEntropyLoss {
  /**
   * Returns the loss = sum over all -(y * ln(a) + (1 - y) * ln(1 - a)).
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @return {number} The loss
   */
  static fn(a, y) {
    const yIsOne = Matrix.mul(
      Matrix.transpose(y),
      Matrix.map(a, (a_i) => {
        // Add small epsilon so if a_i = 0, not doing log(0)
        return Math.log(a_i + EPSILON);
      })
    );
    const yIsZero = Matrix.mul(
      Matrix.transpose(Matrix.map(y, (y_i) => 1 - y_i)),
      Matrix.map(a, (a_i) => {
        // Add small epsilon so if a_i = 1, not doing log(0)
        return Math.log(1 - a_i + EPSILON);
      })
    );
    return -(yIsOne.data[0][0] + yIsZero.data[0][0]);
  }

  /**
   * Returns the output error = [(a - y) / (a * (1 - a))] hadamardProduct outputActivationFunctionDerivative(z).
   * @param {Matrix} z The z of the output layer
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @param {Function} outputActivationFunction The activation function of the output layer
   * @return {Matrix} The output error matrix
   */
  static outputError(z, a, y, outputActivationFunction) {
    const lossDerivativeWRTa = Matrix.sub(a, y).div(
      Matrix.map(a, (a_i) => {
        // Add small epsilon so never divide by zero
        return a_i * (1 - a_i) + EPSILON;
      })
    );
    return lossDerivativeWRTa.mul(outputActivationFunction.derivative(z));
  }
}

/**
 * The cross entropy loss function and output layer error.
 */
export class CrossEntropyLoss {
  /**
   * Returns the loss = sum over all -y * ln(a).
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @return {number} The loss
   */
  static fn(a, y) {
    const loss = Matrix.mul(
      Matrix.transpose(y),
      Matrix.map(a, (a_i) => {
        // Add small epsilon so if a_i = 0, not doing log(0)
        return Math.log(a_i + EPSILON);
      })
    );
    return -loss.data[0][0];
  }

  /**
   * Returns the output error = (-y / a) hadamardProduct outputActivationFunctionDerivative(z).
   * @param {Matrix} z The z of the output layer
   * @param {Matrix} a The activation of the output layer
   * @param {Matrix} y The desired output
   * @param {Function} outputActivationFunction The activation function of the output layer
   * @return {Matrix} The output error matrix
   */
  static outputError(z, a, y, outputActivationFunction) {
    const lossDerivativeWRTa = Matrix.map(y, (y_i) => -y_i).div(
      Matrix.map(a, (a_i) => {
        // Add small epsilon so never divide by zero
        return a_i + EPSILON;
      })
    );
    return lossDerivativeWRTa.mul(outputActivationFunction.derivative(z));
  }
}
