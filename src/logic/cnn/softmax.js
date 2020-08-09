import Matrix from "../matrix";

class Softmax {
  constructor(inputLen, nodes) {
    this.weights = new Matrix(inputLen, nodes).randomizeNormal().div(inputLen);
    this.biases = new Matrix(1, nodes);
  }

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

  backprop(dLdOut, learningRate) {
    // Only one element of dLdOut is nonzero, find it
    for (let i = 0; i < dLdOut.cols; i++) {
      // const gradient = dLdOut.data[0][i];

      // Skip since we only care about nonzero gradient
      // if (gradient === 0) {
      //   continue;
      // }

      // e^z
      const zExp = Matrix.map(this.lastZ, Math.exp);
      const zExpCorrect = zExp.data[0][i];

      // Sum of all e^z
      let S = 0;
      for (let c = 0; c < zExp.cols; c++) {
        S += zExp.data[0][c];
      }

      /* Gradients of output w.r.t z
          = -zExpCorrect * zExp / (S ** 2), for i != correct label 
          = zExpCorrect * (S - zExpCorrect) / (S ** 2), for i == correct label
      */
      const dOutdZ = zExp.mul(-zExpCorrect / S ** 2);
      dOutdZ.data[0][i] = (zExpCorrect * (S - zExpCorrect)) / S ** 2;

      // Gradients of z w.r.t. weights/biases/input, dZdB = 1, so can ignore
      const dZdW = this.lastInput;
      const dZdInputs = this.weights;

      // Gradients of loss w.r.t. z
      // const dLdZ = dOutdZ.mul(gradient);
      const dLdZ = dLdOut.mul(dOutdZ);

      // Gradients of loss w.r.t. weights/biases/input
      const dLdW = Matrix.mul(Matrix.transpose(dZdW), dLdZ);
      const dLdB = dLdZ;
      // const dLdInputs = Matrix.mul(dZdInputs, Matrix.transpose(dLdZ));
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
}

export default Softmax;
