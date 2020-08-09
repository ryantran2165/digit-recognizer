import Matrix from "../matrix";
import Conv from "./conv";
import MaxPool from "./maxpool";
import Softmax from "./softmax";
import { shuffle } from "../utils";

const OUTPUT_NETWORK = true;

class CNN {
  /**
   * Creates a new simple CNN or loads a pretrained model.
   * @param {CNN} cnn The pretrained CNN model
   */
  constructor(cnn = null) {
    if (cnn !== null) {
      this.conv = new Conv(null, null, cnn.conv);
      this.pool = new MaxPool(null, cnn.pool);
      this.softmax = new Softmax(null, null, cnn.softmax);
    } else {
      this.conv = new Conv(8); // 28x28x1 -> 26x26x8
      this.pool = new MaxPool(); // 26x26x8 -> 13x13x8
      this.softmax = new Softmax(13 * 13 * 8, 10); // 13x13x8 -> 10
    }
  }

  /**
   * Passes the image through the CNN.
   * @param {Matrix} image The image as a 2D matrix
   * @return {Array} The result as an array of probabilities
   */
  predict(image) {
    // Out is a row vector of probabilities
    let out = this.conv.forward(image); // 28x28x1 -> 26x26x8
    out = this.pool.forward(out); // 26x26x8 -> 13x13x8
    out = this.softmax.forward(out); // 13x13x8 -> 10
    return out.toArray();
  }

  /**
   * Trains the CNN.
   * @param {Array} trainDatas The array of train datas
   * @param {number} epochs The number of epochs to train for
   * @param {number} learningRate The learning rate
   * @param {Array} testDatas The optional array of test datas
   */
  train(trainDatas, epochs, learningRate, testDatas = null) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      shuffle(trainDatas);

      let loss = 0;
      let numCorrect = 0;

      for (let i = 0; i < trainDatas.length; i++) {
        const trainData = trainDatas[i];
        const image = trainData[0];
        const label = trainData[1];

        // Label is a column vector
        const [_, labelNum, __] = label.max();

        // Forward, out is a row vector
        const [out, l, acc] = this.forward(image, label);

        // Save loss and number correct for printing
        loss += l;
        numCorrect += acc;

        // Initial gradient = dL/dOut = d(-ln(Out))/dOut = -1/Out
        let gradient = new Matrix(1, 10);
        gradient.data[0][labelNum] = -1 / out.data[0][labelNum];

        // Backprop
        gradient = this.softmax.backprop(gradient, learningRate);
        gradient = this.pool.backprop(gradient);
        this.conv.backprop(gradient, learningRate);

        if (i % 100 === 99) {
          console.log(
            "Training: " +
              (i + 1) +
              "/" +
              trainDatas.length +
              " | Average Loss: " +
              loss / 100 +
              " | Accuracy: " +
              numCorrect +
              "/100"
          );
          loss = 0;
          numCorrect = 0;
        }
      }

      if (OUTPUT_NETWORK) {
        console.log(JSON.stringify(this));
      }
    }

    if (testDatas !== null) {
      this.test(testDatas);
    }
  }

  /**
   * Feedforward on one image-label pair.
   * @param {Matrix} image The image to feedforward
   * @param {Matrix} label The correct label as a one-hot column vector
   * @return {Array} The output probabilities as a 1x10 Matrix, the loss, and whether the CNN predicted correctly
   */
  forward(image, label) {
    // Out is a row vector of probabilities
    let out = this.conv.forward(image); // 28x28x1 -> 26x26x8
    out = this.pool.forward(out); // 26x26x8 -> 13x13x8
    out = this.softmax.forward(out); // 13x13x8 -> 10

    // Label is a column vector
    const [_, labelNum, __] = label.max();
    const [___, ____, outMax] = out.max();

    // Calculate cross entropy loss and accuracy
    const loss = -Math.log(out.data[0][labelNum]);
    const acc = outMax === labelNum ? 1 : 0;

    return [out, loss, acc];
  }

  /**
   * Prints the average test loss and accuracy of the CNN on the test set.
   * @param {Array} testDatas The array of test datas
   */
  test(testDatas) {
    let loss = 0;
    let numCorrect = 0;

    // Calculate loss and number correct for all test samples
    for (let testData of testDatas) {
      const image = testData[0];
      const label = testData[1];

      const [_, l, acc] = this.forward(image, label);
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
