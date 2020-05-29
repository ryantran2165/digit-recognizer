import Matrix from "./matrix";
import {
  sigmoid,
  sigmoidDerivative,
  QuadraticCost,
  CrossEntropyCost,
  shuffle,
} from "./helpers";

const CHECK_GRADIENTS = false;
const DEBUG_TRAINING = false;
const OUTPUT_NETWORK = false;

class NeuralNetwork {
  /**
   * Creates a NeuralNetwork with the givens layer sizes or using the preset settings from the optional NeuralNetwork.
   * @param {Array} sizes An array of layer sizes
   * @param {NeuralNetwork} neuralNetwork Optional initial settings
   */
  constructor(sizes, neuralNetwork) {
    if (neuralNetwork) {
      // Deep copy
      this.numLayers = neuralNetwork.sizes.length;
      this.sizes = [];
      for (let size of neuralNetwork.sizes) {
        this.sizes.push(size);
      }

      // Copy bias vectors
      this.biases = [];
      for (let bias of neuralNetwork.biases) {
        this.biases.push(new Matrix(null, null, bias));
      }

      // Copy weight matrices
      this.weights = [];
      for (let weight of neuralNetwork.weights) {
        this.weights.push(new Matrix(null, null, weight));
      }
    } else {
      this.numLayers = sizes.length;
      this.sizes = sizes;

      // Create bias vectors
      this.biases = [];
      for (let i = 1; i < sizes.length; i++) {
        const bias = new Matrix(sizes[i], 1);
        bias.randomizeNormal();
        this.biases.push(bias);
      }

      // Create weight matrices
      this.weights = [];
      for (let i = 1; i < sizes.length; i++) {
        const weight = new Matrix(sizes[i], sizes[i - 1]);
        weight.randomizeNormal();
        this.weights.push(weight);
      }
    }

    this.cost = CrossEntropyCost;
  }

  /**
   * Performs feedforward and returns the result as an array.
   * @param {Matrix} input The input as a Matrix object
   * @return {Array} The result as an array
   */
  feedforward = (input) => {
    let output = input;

    for (let i = 0; i < this.numLayers - 1; i++) {
      const bias = this.biases[i];
      const weight = this.weights[i];

      // a = sigmoid(wx + b)
      output = Matrix.mul(weight, output);
      output.add(bias);
      output.map(sigmoid);
    }

    return output.toArray();
  };

  /**
   * Performs stochastic gradient descent with the specified hyperparameters.
   * @param {Array} trainingDatas The array of training datas
   * @param {number} epochs The number of epochs to train for
   * @param {number} miniBatchSize The size of the mini batches
   * @param {number} learningRate The learning rate
   * @param {number} regularization The regularization parameter
   * @param {Array} testDatas The optional test datas
   */
  stochasticGradientDescent = (
    trainingDatas,
    epochs,
    miniBatchSize,
    learningRate,
    regularization,
    testDatas = null
  ) => {
    // Training datas = [trainingData == [Matrix(input), Matrix(desiredOutput)]]
    const trainingDataSize = trainingDatas.length;

    // Train for specified number of epochs
    for (let i = 0; i < epochs; i++) {
      // Shuffle the training datas every epoch
      shuffle(trainingDatas);

      // Mini batches = [miniBatch == [trainingData == [Matrix(input), Matrix(desiredOutput)]]]
      const miniBatches = [];
      for (let j = 0; j < trainingDataSize; j += miniBatchSize) {
        // Mini batch = [trainingData == [Matrix(input), Matrix(desiredOutput)]]
        const miniBatch = [];

        // Training data = [Matrix(input), Matrix(desiredOutput)]
        for (let k = j; k < j + miniBatchSize; k++) {
          miniBatch.push(trainingDatas[k]);
        }
        miniBatches.push(miniBatch);
      }

      // Update each mini batch
      for (let j = 0; j < miniBatches.length; j++) {
        this.updateMiniBatch(
          miniBatches[j],
          learningRate,
          regularization,
          trainingDataSize
        );

        if (DEBUG_TRAINING) {
          const evaluation = this.evaluate(testDatas);
          console.log(
            "Testing mini batch " +
              (j + 1) +
              "/" +
              miniBatches.length +
              ": " +
              evaluation +
              "/" +
              testDatas.length +
              ", " +
              (100 * evaluation) / testDatas.length +
              "%"
          );
        } else {
          console.log(
            "Finished mini batch " + (j + 1) + "/" + miniBatches.length
          );
        }
      }

      // Testing
      if (testDatas !== null) {
        const evaluation = this.evaluate(testDatas);
        console.log(
          "Testing epoch " +
            (i + 1) +
            "/" +
            epochs +
            ": " +
            evaluation +
            "/" +
            testDatas.length +
            ", " +
            (100 * evaluation) / testDatas.length +
            "%"
        );
      }

      if (OUTPUT_NETWORK) {
        console.log(JSON.stringify(this));
      }
    }
  };

  /**
   * Updates the mini batch by getting the gradient and then applying it.
   * @param {Array} miniBatch The mini batch of training data
   * @param {number} learningRate The learning rate
   * @param {number} regularization The regularization parameter
   * @param {number} trainingDataSize The size of the training data set
   */
  updateMiniBatch = (
    miniBatch,
    learningRate,
    regularization,
    trainingDataSize
  ) => {
    // Cumulative gradients for mini batch
    const biasesGradient = this.createEmptyGradient(this.biases);
    const weightsGradient = this.createEmptyGradient(this.weights);

    // Calculates the cumulative biases and weights gradients for all training data in the mini batch
    for (let trainingData of miniBatch) {
      const input = trainingData[0];
      const desiredOutput = trainingData[1];
      const gradientDelta = this.backpropagate(input, desiredOutput);
      const biasesGradientDelta = gradientDelta[0];
      const weightsGradientDelta = gradientDelta[1];

      // Do gradient checking
      if (CHECK_GRADIENTS) {
        const biasesCheck = this.gradientCheck(
          biasesGradientDelta,
          this.biases,
          input,
          desiredOutput
        );
        console.log("Gradient check biases:", biasesCheck);
        const weightsCheck = this.gradientCheck(
          weightsGradientDelta,
          this.weights,
          input,
          desiredOutput
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
      this.weights[i].mul(
        1 - learningRate * (regularization / trainingDataSize)
      );

      // Weight adjustment by gradient
      this.weights[i].sub(weightsGradient[i].mul(learningRateWithAvg));
    }
  };

  /**
   * Performs backpropagation to calculate the gradient for one training data.
   * @param {Matrix} input The input matrix
   * @param {Matrix} desiredOutput The desired output matrix
   * @return {Array} The array consisting of the biasesGradient and weightsGradient for this one training data
   */
  backpropagate = (input, desiredOutput) => {
    const biasesGradient = this.createEmptyGradient(this.biases);
    const weightsGradient = this.createEmptyGradient(this.weights);

    // Feedforward, store zs and activations by layer
    const zs = [];
    const activations = [input];
    this.trainingFeedforward(zs, activations);

    // Output biasesGradient is simply the output error
    const outputError = this.cost.delta(
      zs[zs.length - 1],
      activations[activations.length - 1],
      desiredOutput
    );
    biasesGradient[biasesGradient.length - 1] = outputError;

    // Output weightsGradient = outputError * beforeOutputActivationTranspose
    weightsGradient[weightsGradient.length - 1] = Matrix.mul(
      outputError,
      Matrix.transpose(activations[activations.length - 2])
    );

    // Backpropagate error to hidden layers
    let hiddenError = outputError;
    for (let i = 2; i < this.numLayers; i++) {
      // Hidden biasesGradient is simply the hidden error
      hiddenError = this.getHiddenError(hiddenError, zs, i);
      biasesGradient[biasesGradient.length - i] = hiddenError;

      // Hidden weightsGradient = hiddenError * beforeHiddenActivationsTranspose
      weightsGradient[weightsGradient.length - i] = Matrix.mul(
        hiddenError,
        Matrix.transpose(activations[activations.length - i - 1])
      );
    }

    return [biasesGradient, weightsGradient];
  };

  /**
   * Feedforward, but also keeping record of the z's and activations per layer.
   * @param {Array} zs The array to store the z records
   * @param {Array} activations The array to store the activation records
   */
  trainingFeedforward = (zs, activations) => {
    for (let i = 0; i < this.numLayers - 1; i++) {
      const bias = this.biases[i];
      const weight = this.weights[i];

      // z = wa + b, a = sigmoid(z)
      const z = Matrix.mul(weight, activations[i]);
      z.add(bias);
      zs.push(z);

      const a = Matrix.map(z, sigmoid);
      activations.push(a);
    }
  };

  /**
   * Calculates the error in the specified hidden layer.
   * hiddenError = (nextWeightsTranspose * nextError) hadamardProduct sigmoidDerivative(z).
   * @param {Matrix} nextError The next layer's error matrix
   * @param {Array} zs The z records
   * @param {number} i The index offset for the biases and weights of his hidden layer
   * @return {Matrix} The hidden error matrix
   */
  getHiddenError = (nextError, zs, i) => {
    const nextWeights = this.weights[this.weights.length - i + 1];
    const nextWeightsTranspose = Matrix.transpose(nextWeights);
    const z = zs[zs.length - i];
    const zSigmoidDerivative = Matrix.map(z, sigmoidDerivative);
    const hiddenError = Matrix.mul(nextWeightsTranspose, nextError);
    hiddenError.mul(zSigmoidDerivative);
    return hiddenError;
  };

  /**
   * Creates an empty gradient with the target array's shape.
   * @param {Array} target The target array
   * @return {Array} The empty gradient array with in the shape of the target array
   */
  createEmptyGradient = (target) => {
    const gradient = [];
    for (let targetMatrix of target) {
      const gradientMatrix = new Matrix(targetMatrix.rows, targetMatrix.cols);
      gradient.push(gradientMatrix);
    }
    return gradient;
  };

  /**
   * Returns a count of how many test cases were passed.
   * @param {Array} testDatas The array of test datas
   * @return {number} The number of test cases passed
   */
  evaluate = (testDatas) => {
    let count = 0;

    for (let testData of testDatas) {
      const input = testData[0];
      const outputArr = this.feedforward(input);
      const desiredOutputArr = testData[1].toArray();
      const outputInteger = outputArr.indexOf(Math.max(...outputArr));
      const desiredOutputInteger = desiredOutputArr.indexOf(
        Math.max(...desiredOutputArr)
      );

      if (outputInteger === desiredOutputInteger) {
        count++;
      }
    }

    return count;
  };

  /**
   * Performs gradient checking technique, manually calculating the gradient using the limit definition the derivative and a small epsilon.
   * @param {Array} gradient The gradient to check
   * @param {Array} target A reference to the neural network's biases or weights
   * @param {Matrix} input The input matrix
   * @param {Matrix} desiredOutput The desired output matrix
   * @return {number} The euclidean norm ratio which should be less than 10^-7
   */
  gradientCheck = (gradient, target, input, desiredOutput) => {
    const desiredArr = desiredOutput.toArray();
    const epsilon = Math.pow(10, -7);
    const gradApprox = this.createEmptyGradient(gradient);

    for (let i = 0; i < gradApprox.length; i++) {
      for (let r = 0; r < gradApprox[i].rows; r++) {
        for (let c = 0; c < gradApprox[i].cols; c++) {
          // Save the original bias or weight value to restore it at the end
          const original = target[i].data[r][c];

          // Calculate the cost with plus epsilon
          target[i].data[r][c] = original + epsilon;
          const outputPlus = this.feedforward(input);
          let costPlus = 0;
          for (let j = 0; j < outputPlus.length; j++) {
            costPlus += 0.5 * Math.pow(desiredArr[j] - outputPlus[j], 2);
          }

          // Calculate the cost with minus epsilon
          target[i].data[r][c] = original - epsilon;
          const outputMinus = this.feedforward(input);
          let costMinus = 0;
          for (let j = 0; j < outputMinus.length; j++) {
            costMinus += 0.5 * Math.pow(desiredArr[j] - outputMinus[j], 2);
          }

          // Limit definition of derivative
          const gradApproxVal = (costPlus - costMinus) / (2 * epsilon);
          gradApprox[i].data[r][c] = gradApproxVal;

          // Restore the initial bias or weight value
          target[i].data[r][c] = original;
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

          paramSum += Math.pow(gradientVal, 2);
          paramApproxSum += Math.pow(gradApproxVal, 2);
          errorSum += Math.pow(gradApproxVal - gradientVal, 2);
        }
      }
    }

    return (
      Math.sqrt(errorSum) / (Math.sqrt(paramApproxSum) + Math.sqrt(paramSum))
    );
  };

  static chooseRegularization = (trainingDatas, crossValDatas, testDatas) => {
    const regularizationOptions = [0.01, 0.03, 0.1, 0.3, 1, 3, 10];
    let bestRegularization = 0.01;
    let bestEvaluation = 0;
    let bestNetwork = null;

    // Train a different model for each regularization option
    for (let i = 0; i < regularizationOptions.length; i++) {
      const curRegularization = regularizationOptions[i];
      const curNetwork = new NeuralNetwork([784, 30, 10]);

      // Train the network using the current regularization setting
      curNetwork.stochasticGradientDescent(
        trainingDatas,
        1,
        10,
        3.0,
        curRegularization
      );

      // Evaluate using cross validation set
      const crossValEvaluation = curNetwork.evaluate(crossValDatas);

      // Choose best neural network based on cross validation set
      if (crossValEvaluation > bestEvaluation) {
        bestRegularization = curRegularization;
        bestEvaluation = crossValEvaluation;
        bestNetwork = curNetwork;
      }
    }

    console.log("Best regularization parameter: " + bestRegularization);
    console.log(
      "Best cross validation set: " +
        bestEvaluation +
        "/" +
        crossValDatas.length +
        ", " +
        (100 * bestEvaluation) / crossValDatas.length +
        "%"
    );

    // Test the generalization of the selected network on the test set
    const generalizationEvaluation = bestNetwork.evaluate(testDatas);
    console.log(
      "Best test set: " +
        generalizationEvaluation +
        "/" +
        testDatas.length +
        ", " +
        (100 * generalizationEvaluation) / testDatas.length +
        "%"
    );

    // Log the best neural network
    console.log("Best network:", JSON.stringify(bestNetwork));
  };
}

export default NeuralNetwork;
