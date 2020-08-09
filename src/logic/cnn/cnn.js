import Matrix from "../matrix";
import Conv from "./conv";
import MaxPool from "./maxpool";
import Softmax from "./softmax";

class CNN {
  constructor() {
    this.conv = new Conv(8); // 28x28x1 -> 26x26x8
    this.pool = new MaxPool(); // 26x26x8 -> 13x13x8
    this.softmax = new Softmax(13 * 13 * 8, 10); // 13x13x8 -> 10
  }

  forward(image, label) {
    // Out is a row vector of probabilities
    let out = this.conv.forward(image);
    out = this.pool.forward(out);
    out = this.softmax.forward(out);

    // Label is a column vector
    const [_, labelNum, __] = label.max();
    const [pred, ___, outMax] = out.max();
    console.log(out);

    // Calculate cross entropy loss and accuracy
    const loss = -Math.log(out.data[0][labelNum]);
    const acc = outMax === labelNum ? 1 : 0;

    return [out, loss, acc, pred];
  }

  train(trainDatas, epochs, learningRate, testDatas = null) {
    for (let i = 0; i < trainDatas.length; i++) {
      const trainData = trainDatas[i];
      const image = trainData[0];
      const label = trainData[1];

      // Label is a column vector
      const [_, labelNum, __] = label.max();

      // Forward, out is a row vector
      const [out, ___, ____, _____] = this.forward(image, label);

      // Initial gradient = dL/dOut = d(-ln(Out))/dOut = -1/Out
      let gradient = new Matrix(1, 10);
      gradient.data[0][labelNum] = -1 / out.data[0][labelNum];

      // Backprop
      gradient = this.softmax.backprop(gradient, learningRate);
      gradient = this.pool.backprop(gradient);
      this.conv.backprop(gradient, learningRate);

      console.log("Training: " + (i + 1) + "/" + trainDatas.length);
    }
    console.log(this);
    this.test(testDatas);
  }

  test(testDatas) {
    let loss = 0;
    let numCorrect = 0;

    // Calculate loss and number correct for all test samples
    for (let testData of testDatas) {
      const image = testData[0];
      const label = testData[1];

      const [_, l, acc, pred] = this.forward(image, label);
      loss += l;
      numCorrect += acc;
    }

    // Print the loss and accuracy
    const numTests = testDatas.length;
    console.log("Test loss:", loss / numTests);
    console.log(
      "Test accuracy: " +
        numCorrect +
        "/" +
        numTests +
        ", " +
        (100 * numCorrect) / numTests +
        "%"
    );
  }
}

export default CNN;
