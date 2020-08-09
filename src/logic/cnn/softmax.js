import Matrix from "../matrix";

class Softmax {
  /**
   * Creates a new softmax layer or loads the pretrained layer.
   * @param {number} inputLen The input layer size
   * @param {number} nodes The output layer size
   * @param {Softmax} softmax The pretrained softmax layer
   */
  constructor(inputLen, nodes, softmax = null) {
    if (softmax !== null) {
      this.weights = new Matrix(null, null, softmax.weights);
      this.biases = new Matrix(null, null, softmax.biases);
    } else {
      this.weights = new Matrix(inputLen, nodes)
        .randomizeNormal()
        .div(inputLen);
      this.biases = new Matrix(1, nodes);
    }
  }

  /**
   * Feedforward on the softmax layer.
   * @param {Array} input The array of input images passed through a max pool layer
   * @return {Matrix} A row vector of final output probabilities
   */
  forward(input) {
    this.lastInputShape = [input.length, input[0].rows, input[0].cols];

    // Flattened row vector
    const flattened = new Matrix(
      1,
      input[0].rows * input[0].cols * input.length
    );
    let flattenedCol = 0;

    // Flatten
    for (let channel of input) {
      for (let r = 0; r < channel.rows; r++) {
        for (let c = 0; c < channel.cols; c++) {
          flattened.data[0][flattenedCol] = channel.data[r][c];
          flattenedCol++;
        }
      }
    }
    this.lastInput = flattened;

    // Product sum and bias
    const z = Matrix.mul(flattened, this.weights).add(this.biases);
    this.lastZ = z;

    // Softmax function
    const exp = Matrix.map(z, Math.exp);
    let sum = 0;
    for (let c = 0; c < exp.cols; c++) {
      sum += exp.data[0][c];
    }
    return exp.div(sum);
  }

  /**
   * Backpropagation on the softmax layer.
   * @param {Matrix} dLdOut Row vector of loss gradients w.r.t. the softmax layer's outputs
   * @param {number} learningRate The learning rate
   * @return {Array} Array of loss gradients w.r.t. the softmax layer's inputs
   */
  backprop(dLdOut, learningRate) {
    // Find correct label index
    let correctI = 0;
    for (let i = 0; i < dLdOut.cols; i++) {
      if (dLdOut.data[0][i] !== 0) {
        correctI = i;
        break;
      }
    }
    const gradient = dLdOut.data[0][correctI];

    // e^z
    const zExp = Matrix.map(this.lastZ, Math.exp);
    const zExpCorrect = zExp.data[0][correctI];

    // Sum of all e^z
    let S = 0;
    for (let c = 0; c < zExp.cols; c++) {
      S += zExp.data[0][c];
    }

    /* Gradients of output w.r.t z
          = -zExpCorrect * zExp / (S ** 2), for wrong label
          = zExpCorrect * (S - zExpCorrect) / (S ** 2), for correct label
      */
    const dOutdZ = zExp.mul(-zExpCorrect / S ** 2);
    dOutdZ.data[0][correctI] = (zExpCorrect * (S - zExpCorrect)) / S ** 2;

    // Gradients of z w.r.t. weights/biases/input, dZdB = 1, so can ignore
    const dZdW = this.lastInput;
    const dZdInputs = this.weights;

    // Gradients of loss w.r.t. z
    const dLdZ = dOutdZ.mul(gradient);

    // Gradients of loss w.r.t. weights/biases/input
    const dLdW = Matrix.mul(Matrix.transpose(dZdW), dLdZ);
    const dLdB = dLdZ;
    const dLdInputs = Matrix.mul(dLdZ, Matrix.transpose(dZdInputs));

    // Update weights and biases
    this.weights.sub(dLdW.mul(learningRate));
    this.biases.sub(dLdB.mul(learningRate));

    // Reshape to original input shape
    const channels = this.lastInputShape[0];
    const rows = this.lastInputShape[1];
    const cols = this.lastInputShape[2];
    let index = 0;
    const dLdInputsReshaped = [];

    for (let i = 0; i < channels; i++) {
      const dLdInputChannel = new Matrix(rows, cols);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          dLdInputChannel.data[r][c] = dLdInputs.data[0][index];
          index++;
        }
      }

      dLdInputsReshaped.push(dLdInputChannel);
    }

    return dLdInputsReshaped;
  }
}

export default Softmax;
