import Matrix from "./matrix";
import { sigmoid, sigmoidDerivative, shuffle } from "./helpers";

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
  }

  /**
   * Performs feedforward and returns the result as an array.
   * @param {Matrix} input The input as a Matrix object
   * @return {Array} The result as an array
   */
  feedforward = input => {
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
   * @param {Array} testDatas The optional test datas
   */
  stochasticGradientDescent = (
    trainingDatas,
    epochs,
    miniBatchSize,
    learningRate,
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
      for (let miniBatch of miniBatches) {
        this.updateMiniBatch(miniBatch, learningRate);
      }

      // Testing
      if (testDatas !== null) {
        console.log(
          "Epoch " +
            i +
            ": " +
            this.evaluate(testDatas) +
            " / " +
            testDatas.length
        );
      }
    }
  };

  /**
   * Updates the mini batch by getting the gradient and then applying it.
   * @param {Array} miniBatch The mini batch of training data
   * @param {number} learningRate The learning rate
   */
  updateMiniBatch = (miniBatch, learningRate) => {
    // Cumulative gradients for mini batch
    const biasesGradient = this.createEmptyGradient(this.biases);
    const weightsGradient = this.createEmptyGradient(this.weights);
    this.getMiniBatchGradient(miniBatch, biasesGradient, weightsGradient);
    this.applyMiniBatchGradient(
      biasesGradient,
      weightsGradient,
      learningRate,
      miniBatch.length
    );
  };

  /**
   * Calculates the cumulative biases and weights gradients for all training data in the mini batch.
   * @param {Array} miniBatch The mini batch of training data
   * @param {Array} biasesGradient The array representing the cumulative biases gradient
   * @param {Array} weightsGradient The array representing the cumulative weights gradient
   */
  getMiniBatchGradient = (miniBatch, biasesGradient, weightsGradient) => {
    for (let trainingData of miniBatch) {
      const input = trainingData[0];
      const desiredOutput = trainingData[1];
      const gradientDelta = this.backpropagate(input, desiredOutput);
      const biasesGradientDelta = gradientDelta[0];
      const weightsGradientDelta = gradientDelta[1];

      // this.gradientCheck(
      //   biasesGradientDelta,
      //   this.biases,
      //   input,
      //   desiredOutput
      // );
      // this.gradientCheck(
      //   weightsGradientDelta,
      //   this.weights,
      //   input,
      //   desiredOutput
      // );

      // Add gradient delta to miniBatch
      for (let i = 0; i < this.numLayers - 1; i++) {
        const biasGradient = biasesGradient[i];
        const weightGradient = weightsGradient[i];
        const biasGradientDelta = biasesGradientDelta[i];
        const weightGradientDelta = weightsGradientDelta[i];

        biasGradient.add(biasGradientDelta);
        weightGradient.add(weightGradientDelta);
      }
    }
  };

  /**
   * Apply the cumulative biases and weights gradients to the network's biases and weights.
   * @param {Array} biasesGradient The array representing the cumulative biases gradient
   * @param {Array} weightsGradient The array representing the cumulative weights gradient
   * @param {number} learningRate The learning rate
   * @param {Array} miniBatch The mini batch of training data
   */
  applyMiniBatchGradient = (
    biasesGradient,
    weightsGradient,
    learningRate,
    miniBatchSize
  ) => {
    for (let i = 0; i < this.numLayers - 1; i++) {
      const bias = this.biases[i];
      const weight = this.weights[i];
      const biasGradient = biasesGradient[i];
      const weightGradient = weightsGradient[i];

      // Adding the negative (i.e. subtracting) average gradient for mini batch
      bias.sub(biasGradient.mul(learningRate / miniBatchSize));
      weight.sub(weightGradient.mul(learningRate / miniBatchSize));
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

    // Output biasesGradient is simply the output delta
    const outputError = this.getOutputError(zs, activations, desiredOutput);
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
   * Creates an empty gradient with the target array's shape.
   * @param {Array} target The target array
   * @return {Array} The empty gradient array with in the shape of the target array
   */
  createEmptyGradient = target => {
    const gradient = [];
    for (let targetMatrix of target) {
      const gradientMatrix = new Matrix(targetMatrix.rows, targetMatrix.cols);
      gradient.push(gradientMatrix);
    }
    return gradient;
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
   * Calculates the error in the output layer.
   * outputError = (costDerivative == outputActivation - desiredOutput) hadamardProduct outputZSigmoidDerivative.
   * @param {Array} zs The z records
   * @param {Array} activations The activation records
   * @param {Matrix} desiredOutput The desired output matrix
   * @return {Matrix} The output error matrix
   */
  getOutputError = (zs, activations, desiredOutput) => {
    const outputActivation = activations[activations.length - 1];
    const costDerivative = Matrix.sub(outputActivation, desiredOutput);
    const outputZ = zs[zs.length - 1];
    const outputZSigmoidDerivative = Matrix.map(outputZ, sigmoidDerivative);
    const outputError = costDerivative.mul(outputZSigmoidDerivative);
    return outputError;
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
   * Returns a count of how many test cases were passed.
   * @param {Array} testDatas The array of test datas
   * @return {number} The number of test cases passed
   */
  evaluate = testDatas => {
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

    let sum = 0;
    let paramSum = 0;
    let paramApproxSum = 0;

    // Sum all Euclidean components
    for (let i = 0; i < gradApprox.length; i++) {
      for (let r = 0; r < gradApprox[i].rows; r++) {
        for (let c = 0; c < gradApprox[i].cols; c++) {
          const gradientVal = gradient[i].data[r][c];
          const gradApproxVal = gradApprox[i].data[r][c];

          paramSum += Math.pow(gradientVal, 2);
          paramApproxSum += Math.pow(gradApproxVal, 2);
          sum += Math.pow(gradApproxVal - gradientVal, 2);
        }
      }
    }

    // Euclidean norm
    sum = Math.sqrt(sum);
    sum /= Math.sqrt(paramApproxSum) + Math.sqrt(paramSum);

    // Should be less than 10^-7
    console.log("Sum", sum);
  };
}

export default NeuralNetwork;
