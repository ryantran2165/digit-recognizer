import Matrix from "../matrix";
import {
  SigmoidActivation,
  ReLUActivation,
  LeakyReLUActivation,
  SoftmaxActivation,
  QuadraticLoss,
  BinaryCrossEntropyLoss,
  CrossEntropyLoss,
} from "./ffnn-helpers";
import { shuffle } from "../utils";

const CHECK_GRADIENTS = false;
const LOG_MINI_BATCH_ACCURACY = false;
const LOG_MINI_BATCH_COST = false;
const OUTPUT_NETWORK = true;

class FFNN {
  /**
   * Creates a new FFNN with the givens layer sizes or loads a pretrained model.
   * @param {Array} sizes An array of layer sizes
   * @param {FFNN} ffnn Optional initial settings
   */
  constructor(sizes, ffnn) {
    this.hiddenActivationFunction = ReLUActivation;
    this.outputActivationFunction = SoftmaxActivation;
    this.lossFunction = BinaryCrossEntropyLoss;

    if (ffnn) {
      // Deep copy
      this.numLayers = ffnn.sizes.length;
      this.sizes = [];
      for (let size of ffnn.sizes) {
        this.sizes.push(size);
      }

      // Copy bias vectors
      this.biases = [];
      for (let bias of ffnn.biases) {
        this.biases.push(new Matrix(null, null, bias));
      }

      // Copy weight matrices
      this.weights = [];
      for (let weight of ffnn.weights) {
        this.weights.push(new Matrix(null, null, weight));
      }

      // Save mean and std in case of standardization
      if (ffnn.hasOwnProperty("trainMean")) {
        this.trainMean = new Matrix(null, null, ffnn.trainMean);
        this.trainSTD = new Matrix(null, null, ffnn.trainSTD);
      }
    } else {
      this.numLayers = sizes.length;
      this.sizes = sizes;

      // Create bias vectors
      this.biases = [];
      for (let i = 1; i < sizes.length; i++) {
        const bias = new Matrix(sizes[i], 1);

        if (this.hiddenActivationFunction === ReLUActivation) {
          // He initialization, biases = 0
          bias.map((b) => 0);
        } else {
          bias.randomizeNormal();
        }

        this.biases.push(bias);
      }

      // Create weight matrices
      this.weights = [];
      for (let i = 1; i < sizes.length; i++) {
        const weight = new Matrix(sizes[i], sizes[i - 1]);

        // He initialization, weights random standard normal * sqrt(2 / # incoming connections)
        weight.randomizeNormal();
        if (this.hiddenActivationFunction === ReLUActivation) {
          weight.map((w) => w * Math.sqrt(2 / sizes[i - 1]));
        }

        this.weights.push(weight);
      }
    }
  }

  /**
   * Performs feedforward and returns the result as an array.
   * @param {Matrix} input The input as a Matrix object (vector)
   * @return {Array} The result as an array
   */
  predict(input) {
    let output = input;

    for (let i = 0; i < this.numLayers - 1; i++) {
      const bias = this.biases[i];
      const weight = this.weights[i];

      // z = wx + b, a = activationFunction(z)
      const z = Matrix.mul(weight, output);
      z.add(bias);
      output =
        i === this.numLayers - 2
          ? this.outputActivationFunction.fn(z)
          : this.hiddenActivationFunction.fn(z);
    }

    return output.toArray();
  }

  /**
   * Performs stochastic gradient descent with the specified hyperparameters.
   * @param {Array} trainDatas The array of train datas
   * @param {number} epochs The number of epochs to train for
   * @param {number} miniBatchSize The size of the mini batches
   * @param {number} learningRate The learning rate
   * @param {number} regularization The regularization parameter
   * @param {Array} testDatas The optional test datas
   */
  stochasticGradientDescent(
    trainDatas,
    epochs,
    miniBatchSize,
    learningRate,
    regularization,
    testDatas = null
  ) {
    // Train datas = [trainData == [Matrix(input), Matrix(targetOutput)]]
    const trainDataSize = trainDatas.length;

    // Train for specified number of epochs
    for (let i = 0; i < epochs; i++) {
      // Shuffle the train datas every epoch
      shuffle(trainDatas);

      // Mini batches = [miniBatch == [trainData == [Matrix(input), Matrix(targetOutput)]]]
      const miniBatches = [];
      for (let j = 0; j < trainDataSize; j += miniBatchSize) {
        // Mini batch = [trainData == [Matrix(input), Matrix(targetOutput)]]
        const miniBatch = [];

        // Train data = [Matrix(input), Matrix(targetOutput)]
        for (let k = j; k < j + miniBatchSize; k++) {
          miniBatch.push(trainDatas[k]);
        }
        miniBatches.push(miniBatch);
      }

      // Update each mini batch
      for (let j = 0; j < miniBatches.length; j++) {
        this.updateMiniBatch(
          miniBatches[j],
          learningRate,
          regularization,
          trainDataSize
        );

        // Accuracy on test set
        if (LOG_MINI_BATCH_ACCURACY) {
          const accuracy = this.accuracy(testDatas);
          console.log(
            "Testing mini batch " +
              (j + 1) +
              "/" +
              miniBatches.length +
              ": " +
              accuracy +
              "/" +
              testDatas.length +
              ", " +
              (100 * accuracy) / testDatas.length +
              "%"
          );
        } else {
          // Only log this if not doing the more detailed accuracy logging
          console.log(
            "Finished mini batch " + (j + 1) + "/" + miniBatches.length
          );
        }

        // Cost on train set
        if (LOG_MINI_BATCH_COST) {
          const trainCost = this.trainCost(trainDatas, regularization);
          console.log(
            "Train cost: " +
              trainCost[0] +
              ", " +
              trainCost[1] +
              "/" +
              trainDatas.length +
              ", " +
              (100 * trainCost[1]) / trainDatas.length +
              "%"
          );
        }
      }

      // Testing
      if (testDatas !== null) {
        const accuracy = this.accuracy(testDatas);
        console.log(
          "Testing epoch " +
            (i + 1) +
            "/" +
            epochs +
            ": " +
            accuracy +
            "/" +
            testDatas.length +
            ", " +
            (100 * accuracy) / testDatas.length +
            "%"
        );
      }

      if (OUTPUT_NETWORK) {
        console.log(JSON.stringify(this));
      }
    }
  }

  /**
   * Updates the mini batch by getting the gradient and then applying it.
   * @param {Array} miniBatch The mini batch of train data
   * @param {number} learningRate The learning rate
   * @param {number} regularization The regularization parameter
   * @param {number} trainDataSize The size of the train data set
   */
  updateMiniBatch(miniBatch, learningRate, regularization, trainDataSize) {
    // Cumulative gradients for mini batch
    const biasesGradient = this.createEmptyGradient(this.biases);
    const weightsGradient = this.createEmptyGradient(this.weights);

    // Calculates the cumulative biases and weights gradients for all train data in the mini batch
    for (let trainData of miniBatch) {
      const input = trainData[0];
      const targetOutput = trainData[1];
      const gradientDelta = this.backprop(input, targetOutput);
      const biasesGradientDelta = gradientDelta[0];
      const weightsGradientDelta = gradientDelta[1];

      // Do gradient checking
      if (CHECK_GRADIENTS) {
        const biasesCheck = this.gradientCheck(
          biasesGradientDelta,
          this.biases,
          input,
          targetOutput
        );
        console.log("Gradient check biases:", biasesCheck);
        const weightsCheck = this.gradientCheck(
          weightsGradientDelta,
          this.weights,
          input,
          targetOutput
        );
        console.log("Gradient check weights:", weightsCheck);
      }

      // Add gradient delta to miniBatch
      for (let i = 0; i < this.numLayers - 1; i++) {
        biasesGradient[i].add(biasesGradientDelta[i]);
        weightsGradient[i].add(weightsGradientDelta[i]);
      }
    }

    // Apply the cumulative biases and weights gradients to the network's biases and weights
    for (let i = 0; i < this.numLayers - 1; i++) {
      const learningRateWithAvg = learningRate / miniBatch.length;

      // Bias adjustment by gradient, no regularization
      this.biases[i].sub(biasesGradient[i].mul(learningRateWithAvg));

      // Weight regularization
      this.weights[i].mul(1 - learningRate * (regularization / trainDataSize));

      // Weight adjustment by gradient
      this.weights[i].sub(weightsGradient[i].mul(learningRateWithAvg));
    }
  }

  /**
   * Performs backpropagation to calculate the gradient for one train data.
   * @param {Matrix} input The input matrix
   * @param {Matrix} targetOutput The target output matrix
   * @return {Array} The array consisting of the biasesGradient and weightsGradient for this one train data
   */
  backprop(input, targetOutput) {
    const biasesGradient = this.createEmptyGradient(this.biases);
    const weightsGradient = this.createEmptyGradient(this.weights);

    // Feedforward, store zs and activations by layer
    const zs = [];
    const activations = [input];
    this.forward(zs, activations);

    // Output error = lossDerivativeWRTa hadamardProduct outputActivationFunctionDerivative(outputZ)
    const outputError = this.lossFunction.outputError(
      zs[zs.length - 1],
      activations[activations.length - 1],
      targetOutput,
      this.outputActivationFunction
    );

    // Output biasesGradient is simply the output error
    biasesGradient[biasesGradient.length - 1] = outputError;

    // Output weightsGradient = outputError * beforeOutputActivationTranspose
    weightsGradient[weightsGradient.length - 1] = Matrix.mul(
      outputError,
      Matrix.transpose(activations[activations.length - 2])
    );

    // Backpropagate error to hidden layers
    let hiddenError = outputError;
    for (let i = 2; i < this.numLayers; i++) {
      // Hidden error = (nextWeightsTranspose * nextError which is last pass's hidden error) hadamardProduct hiddenActivationFunctionDerivative(z)
      const nextWeightsTranspose = Matrix.transpose(
        this.weights[this.weights.length - i + 1]
      );
      const hiddenActivationFunctionDerivative = this.hiddenActivationFunction.derivative(
        zs[zs.length - i]
      );
      hiddenError = Matrix.mul(nextWeightsTranspose, hiddenError).mul(
        hiddenActivationFunctionDerivative
      );

      // Hidden biasesGradient is simply the hidden error
      biasesGradient[biasesGradient.length - i] = hiddenError;

      // Hidden weightsGradient = hiddenError * beforeHiddenActivationsTranspose
      weightsGradient[weightsGradient.length - i] = Matrix.mul(
        hiddenError,
        Matrix.transpose(activations[activations.length - i - 1])
      );
    }

    return [biasesGradient, weightsGradient];
  }

  /**
   * Feedforward, but also keeping record of the z's and activations per layer.
   * @param {Array} zs The array to store the z records
   * @param {Array} activations The array to store the activation records
   */
  forward(zs, activations) {
    for (let i = 0; i < this.numLayers - 1; i++) {
      const bias = this.biases[i];
      const weight = this.weights[i];

      // z = wa + b, a = activationFunction(z)
      const z = Matrix.mul(weight, activations[i]);
      z.add(bias);
      zs.push(z);

      const a =
        i === this.numLayers - 2
          ? this.outputActivationFunction.fn(z)
          : this.hiddenActivationFunction.fn(z);
      activations.push(a);
    }
  }

  /**
   * Creates an empty gradient with the target array's shape.
   * @param {Array} arr The target array
   * @return {Array} The empty gradient array with in the shape of the target array
   */
  createEmptyGradient(arr) {
    const gradient = [];
    for (let targetMatrix of arr) {
      const gradientMatrix = new Matrix(targetMatrix.rows, targetMatrix.cols);
      gradient.push(gradientMatrix);
    }
    return gradient;
  }

  /**
   * Returns a count of how many test cases were passed.
   * @param {Array} testDatas The array of test datas
   * @return {number} The number of test cases passed
   */
  accuracy(testDatas) {
    let count = 0;

    for (let testData of testDatas) {
      const input = testData[0];
      const outputArr = this.predict(input);
      const targetOutputArr = testData[1].toArray();
      const outputInteger = outputArr.indexOf(Math.max(...outputArr));
      const targetOutputInteger = targetOutputArr.indexOf(
        Math.max(...targetOutputArr)
      );

      // Count number correct
      if (outputInteger === targetOutputInteger) {
        count++;
      }
    }

    return count;
  }

  /**
   * Returns the train cost and correct count for the data set using the regularization parameter.
   * @param {Array} datas The array of data to get the train cost for
   * @param {number} regularization The regularization parameter
   * @return {Array} The train cost and correct count
   */
  trainCost(datas, regularization) {
    let cost = 0;
    let count = 0;

    // Add loss of each data point
    for (let data of datas) {
      const input = data[0];
      const targetOutput = data[1];

      const outputArr = this.predict(input);
      const targetOutputArr = targetOutput.toArray();
      const outputInteger = outputArr.indexOf(Math.max(...outputArr));
      const targetOutputInteger = targetOutputArr.indexOf(
        Math.max(...targetOutputArr)
      );

      // Count correct
      if (outputInteger === targetOutputInteger) {
        count++;
      }

      // Output cost
      cost +=
        this.lossFunction.fn(Matrix.vectorFromArray(outputArr), targetOutput) /
        datas.length;

      // Regularization cost
      let squaredWeights = 0;
      for (let weight of this.weights) {
        for (let r = 0; r < weight.rows; r++) {
          for (let c = 0; c < weight.cols; c++) {
            squaredWeights += weight.data[r][c] ** 2;
          }
        }
      }
      cost += 0.5 * (regularization / datas.length) * squaredWeights;
    }

    return [cost, count];
  }

  /**
   * Performs gradient checking technique, manually calculating the gradient using the limit definition the derivative and a small epsilon.
   * @param {Array} gradient The gradient to check
   * @param {Array} weightOrBias A reference to the neural network's biases or weights
   * @param {Matrix} input The input matrix
   * @param {Matrix} targetOutput The targed output matrix
   * @return {number} The euclidean norm ratio which should be less than 10^-7
   */
  gradientCheck(gradient, weightOrBias, input, targetOutput) {
    const targetOutputArr = targetOutput.toArray();
    const epsilon = 10 ** -7;
    const gradApprox = this.createEmptyGradient(gradient);

    for (let i = 0; i < gradApprox.length; i++) {
      for (let r = 0; r < gradApprox[i].rows; r++) {
        for (let c = 0; c < gradApprox[i].cols; c++) {
          // Save the original bias or weight value to restore it at the end
          const original = weightOrBias[i].data[r][c];

          // Calculate the loss with plus epsilon
          weightOrBias[i].data[r][c] = original + epsilon;
          const outputPlus = this.predict(input);
          let lossPlus = 0;
          for (let j = 0; j < outputPlus.length; j++) {
            lossPlus += 0.5 * (targetOutputArr[j] - outputPlus[j]) ** 2;
          }

          // Calculate the loss with minus epsilon
          weightOrBias[i].data[r][c] = original - epsilon;
          const outputMinus = this.predict(input);
          let lossMinus = 0;
          for (let j = 0; j < outputMinus.length; j++) {
            lossMinus += 0.5 * (targetOutputArr[j] - outputMinus[j]) ** 2;
          }

          // Limit definition of derivative
          const gradApproxVal = (lossPlus - lossMinus) / (2 * epsilon);
          gradApprox[i].data[r][c] = gradApproxVal;

          // Restore the initial bias or weight value
          weightOrBias[i].data[r][c] = original;
        }
      }
    }

    let paramSum = 0;
    let paramApproxSum = 0;
    let errorSum = 0;

    // Sum all Euclidean components
    for (let i = 0; i < gradApprox.length; i++) {
      for (let r = 0; r < gradApprox[i].rows; r++) {
        for (let c = 0; c < gradApprox[i].cols; c++) {
          const gradientVal = gradient[i].data[r][c];
          const gradApproxVal = gradApprox[i].data[r][c];

          paramSum += gradientVal ** 2;
          paramApproxSum += gradApproxVal ** 2;
          errorSum += (gradApproxVal - gradientVal) ** 2;
        }
      }
    }

    return (
      Math.sqrt(errorSum) / (Math.sqrt(paramApproxSum) + Math.sqrt(paramSum))
    );
  }

  /**
   * Automation for choosing the regularization parameter.
   * @param {Array} trainDatas The array of train datas
   * @param {Array} valDatas The array of validation datas
   * @param {Array} testDatas The array of test datas
   */
  static chooseHypeparameters(trainDatas, valDatas, testDatas) {
    const miniBatchOptions = [1, 10, 20, 50, 100];
    const learningRateOptions = [0.01, 0.03, 0.1, 0.3, 1, 3, 10];
    const regularizationOptions = [0.01, 0.03, 0.1, 0.3, 1, 3, 10];
    let bestMiniBatch = 1;
    let bestLearningRate = 0.01;
    let bestRegularization = 0.01;
    let bestAccuracy = 0;
    let bestNetwork = null;

    // Train a different model for each combination of options
    for (let i = 0; i < miniBatchOptions.length; i++) {
      for (let j = 0; j < learningRateOptions.length; j++) {
        for (let k = 0; k < regularizationOptions.length; k++) {
          const curMiniBatch = miniBatchOptions[i];
          const curLearningRate = learningRateOptions[j];
          const curRegularization = regularizationOptions[k];

          // Train the network using the current settings
          const curNetwork = new FFNN([784, 30, 10]);
          curNetwork.stochasticGradientDescent(
            trainDatas,
            1,
            curMiniBatch,
            curLearningRate,
            curRegularization
          );

          // Evaluate accuracy using validation set
          const valAccuracy = curNetwork.accuracy(valDatas);

          // Choose best neural network based on validation set
          if (valAccuracy > bestAccuracy) {
            bestMiniBatch = curMiniBatch;
            bestLearningRate = curLearningRate;
            bestRegularization = curRegularization;
            bestAccuracy = valAccuracy;
            bestNetwork = curNetwork;
          }
        }
      }
    }

    console.log("Best mini batch size: " + bestMiniBatch);
    console.log("Best learning rate: " + bestLearningRate);
    console.log("Best regularization: " + bestRegularization);
    console.log(
      "Best validation set: " +
        bestAccuracy +
        "/" +
        valDatas.length +
        ", " +
        (100 * bestAccuracy) / valDatas.length +
        "%"
    );

    // Test the generalization of the selected network on the test set
    const generalizationAccuracy = bestNetwork.accuracy(testDatas);
    console.log(
      "Best test set: " +
        generalizationAccuracy +
        "/" +
        testDatas.length +
        ", " +
        (100 * generalizationAccuracy) / testDatas.length +
        "%"
    );

    // Log the best neural network
    console.log("Best network:", JSON.stringify(bestNetwork));
  }
}

export default FFNN;
