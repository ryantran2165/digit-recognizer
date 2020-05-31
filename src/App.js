import React, { Component } from "react";
import Title from "./components/title";
import Description from "./components/description";
import Sketch from "./components/sketch";
import Button from "./components/button";
import RangeInput from "./components/range-input";
import Label from "./components/label";
import NeuralNetwork from "./logic/neural-network";
import Matrix from "./logic/matrix";
import loadMNIST from "./logic/mnist";
import neuralNetworkPretrained from "./neural-network-pretrained.json";

const OUTPUT_MNIST = false;
const NUM_TRAINING = 50000;
const NUM_CROSS_VAL = 10000;
const NUM_TESTING = 10000;
const STANDARDIZATION_OVER_NORMALIZATION = true;
const EPSILON = 10 ** -100;
const INCLUDE_CROSS_VAL_SET = false;

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      epochs: 0,
      guess: "",
      pixels: Array(784).fill(0),
      clearRequested: false,
      isTraining: false,
      usePretrained: true,
      outputArr: [],
      trainingSize: 5000,
    };

    this.dataFormatted = false;
    loadMNIST((data) => {
      this.mnist = data;
      this.neuralNetwork = new NeuralNetwork(null, neuralNetworkPretrained);
      console.log("All files loaded");

      if (OUTPUT_MNIST) {
        console.log(this.mnist);
      }
    });
  }

  handleClickTrain = () => {
    if (this.neuralNetwork !== undefined) {
      this.setState({ isTraining: true });

      this.timer = setTimeout(() => {
        // Only need to format if the current setting disagrees OR it's the first time formatting
        if (
          this.neuralNetwork.standardizationOverNormalization !==
            STANDARDIZATION_OVER_NORMALIZATION ||
          !this.alreadyFormatted
        ) {
          this.formatData();
          if (STANDARDIZATION_OVER_NORMALIZATION) {
            this.standardizeData();
          } else {
            this.normalizeData();
          }
          this.alreadyFormatted = true;
        }

        console.log("Starting training");
        this.neuralNetwork.stochasticGradientDescent(
          this.trainingDatas.slice(0, this.state.trainingSize),
          1,
          10,
          0.03,
          1.0,
          this.testDatas
        );

        // NeuralNetwork.chooseHypeparameters(
        //   this.trainingDatas,
        //   this.crossValDatas,
        //   this.testDatas
        // );

        this.setState((prevState) => ({
          epochs: prevState.epochs + 1,
          isTraining: false,
        }));

        clearTimeout(this.timer);
      }, 250);
    }
  };

  formatData() {
    console.log("Formatting training data");
    this.trainingDatas = this.loadDatas(
      this.mnist.trainingImages.slice(0, NUM_TRAINING),
      this.mnist.trainingLabels.slice(0, NUM_TRAINING)
    );

    if (INCLUDE_CROSS_VAL_SET) {
      console.log("Formatting cross validation data");
      this.crossValDatas = this.loadDatas(
        this.mnist.trainingImages.slice(
          NUM_TRAINING,
          NUM_TRAINING + NUM_CROSS_VAL
        ),
        this.mnist.trainingLabels.slice(
          NUM_TRAINING,
          NUM_TRAINING + NUM_CROSS_VAL
        )
      );
    }

    console.log("Formatting testing data");
    this.testDatas = this.loadDatas(
      this.mnist.testImages.slice(0, NUM_TESTING),
      this.mnist.testLabels.slice(0, NUM_TESTING)
    );
  }

  loadDatas = (images, labels) => {
    const datas = [];

    for (let i = 0; i < images.length; i++) {
      const inputArr = images[i];
      const desiredInteger = labels[i];
      const desiredArr = [];

      for (let j = 0; j < 10; j++) {
        if (desiredInteger === j) {
          desiredArr.push(1);
        } else {
          desiredArr.push(0);
        }
      }

      datas.push([
        Matrix.vectorFromArray(inputArr),
        Matrix.vectorFromArray(desiredArr),
      ]);
    }

    return datas;
  };

  standardizeData = () => {
    // Neural network does NOT have predefined training mean and STD, calculate them and use that to standardize everything
    if (!this.neuralNetwork.hasOwnProperty("trainingMean")) {
      const mean = new Matrix(784, 1); // mean = sum(pixels) / numTraining
      const std = new Matrix(784, 1); // std = sqrt(sum((pixels - mean)^2) / numTraining)

      // Calculate sum(pixels)
      console.log("Calculating training data mean");
      for (let trainingData of this.trainingDatas) {
        mean.add(trainingData[0]);
      }

      // Divide by total number of training datas
      mean.map((x) => x / this.trainingDatas.length);

      // Calculate sum((pixels - mean)^2)
      console.log("Calculating training data standard deviation");
      for (let trainingData of this.trainingDatas) {
        std.add(Matrix.sub(trainingData[0], mean).map((x) => x ** 2));
      }

      // Divide by total number of training datas and square root
      std.map((x) => Math.sqrt(x / this.trainingDatas.length));

      // Save to neural network
      this.neuralNetwork.trainingMean = mean;
      this.neuralNetwork.trainingSTD = std;
    }

    // Retrieve mean and std from neural network
    const mean = this.neuralNetwork.trainingMean;
    const std = this.neuralNetwork.trainingSTD;

    // Standardize training images, add small epsilon so never divide by zero
    console.log("Standardizing training data");
    for (let i = 0; i < this.trainingDatas.length; i++) {
      this.trainingDatas[i][0]
        .sub(mean)
        .div(Matrix.map(std, (x) => x + EPSILON));
    }

    if (INCLUDE_CROSS_VAL_SET) {
      // Standardize validation images, add small epsilon so never divide by zero
      console.log("Standardizing cross validation data");
      for (let i = 0; i < this.crossValDatas.length; i++) {
        this.crossValDatas[i][0]
          .sub(mean)
          .div(Matrix.map(std, (x) => x + EPSILON));
      }
    }

    // Standardize test images, add small epsilon so never divide by zero
    console.log("Standardizing testing data");
    for (let i = 0; i < this.testDatas.length; i++) {
      this.testDatas[i][0].sub(mean).div(Matrix.map(std, (x) => x + EPSILON));
    }

    this.neuralNetwork.standardizationOverNormalization = true;
  };

  normalizeData = () => {
    // Normalize training images
    for (let trainingData of this.trainingDatas) {
      trainingData[0].div(255);
    }

    if (INCLUDE_CROSS_VAL_SET) {
      // Normalize validation images
      for (let crossValData of this.crossValDatas) {
        crossValData[0].div(255);
      }
    }

    // Normalize test images
    for (let testData of this.testDatas) {
      testData[0].div(255);
    }

    this.neuralNetwork.standardizationOverNormalization = false;
  };

  handleClickGuess = () => {
    if (this.neuralNetwork !== undefined) {
      const input = Matrix.vectorFromArray(this.state.pixels);

      // Standardize or normalize input pixels
      if (this.neuralNetwork.standardizationOverNormalization) {
        input
          .sub(this.neuralNetwork.trainingMean)
          .div(Matrix.map(this.neuralNetwork.trainingSTD, (x) => x + EPSILON));
      } else {
        input.div(255);
      }

      const outputArr = this.neuralNetwork.feedforward(input);
      const guess = outputArr.indexOf(Math.max(...outputArr));

      // Update state about guess and displayed probabilities
      this.setState({
        guess,
        outputArr,
      });
    }
  };

  handleClickClear = () => {
    this.setState({ clearRequested: true });
  };

  handleClear = () => {
    this.setState({
      pixels: Array(784).fill(0),
      clearRequested: false,
      guess: "",
      outputArr: [],
    });
  };

  handleDraw = (pixels) => {
    this.setState({
      pixels,
    });
  };

  handleSwitchChange = () => {
    this.setState(
      (prevState) => ({ usePretrained: !prevState.usePretrained }),
      () => {
        this.neuralNetwork = this.state.usePretrained
          ? new NeuralNetwork(null, neuralNetworkPretrained)
          : new NeuralNetwork([784, 30, 10]);
        this.setState({ epochs: 0 });
      }
    );
  };

  updateTrainingSize = (e) => {
    this.setState({ trainingSize: Number(e.target.value) });
  };

  render() {
    const probabilities = this.state.outputArr.map((probability, number) => (
      <h5 key={number}>
        {number}: {(probability * 100).toFixed(1)}%
      </h5>
    ));

    return (
      <div className="App container text-center pt-5">
        <div className="row">
          <div className="col">
            <Title text="Handwritten Digits Neural Network" />
          </div>
        </div>
        <div className="row">
          <div className="col">
            <Description
              text={
                "Recognizes handwritten digits using a neural network.\nDraw a number from 0 to 9 and have the neural network guess what you drew.\nTry to center and scale your drawing to fill the majority of the canvas.\nYou can train the neural network with the specified training size for an epoch.\n*Open the console for training progress.\nOr you can just use the pre-trained neural network."
              }
            />
          </div>
        </div>
        <div className="row justify-content-center pt-3">
          <div className="col custom-control custom-switch">
            <input
              type="checkbox"
              className="custom-control-input"
              id="customSwitch"
              checked={this.state.usePretrained}
              onChange={this.handleSwitchChange}
            />
            <label className="custom-control-label" htmlFor="customSwitch">
              Use pre-trained neural network
            </label>
          </div>
        </div>
        <div className="row justify-content-center pt-3">
          <div className="col col-lg-4">
            <div className="row">
              <div className="col">
                <RangeInput
                  min={1000}
                  max={50000}
                  step={1000}
                  defaultValue={this.state.trainingSize}
                  id="trainingSize"
                  onChange={this.updateTrainingSize}
                />
              </div>
            </div>
            <div className="row">
              <div className="col">
                <Label text="Training size" value={this.state.trainingSize} />
              </div>
            </div>
          </div>
        </div>
        <div className="row justify-content-center pt-3">
          <div className="col-4 col-md-3 col-xl-2">
            <Button
              value="Train"
              loadingValue="Training..."
              isLoading={this.state.isTraining}
              onClick={this.handleClickTrain}
            />
          </div>
          <div className="col-4 col-md-3 col-xl-2">
            <Button value="Guess" onClick={this.handleClickGuess} />
          </div>
          <div className="col-4 col-md-3 col-xl-2">
            <Button value="Clear" onClick={this.handleClickClear} />
          </div>
        </div>
        <div className="row pt-3">
          <div className="col">
            <Sketch
              width={500}
              height={500}
              onDraw={(pixels) => this.handleDraw(pixels)}
              clearRequested={this.state.clearRequested}
              onClear={this.handleClear}
            />
          </div>
        </div>
        <div className="row justify-content-center">
          <div className="col-6 col-lg-4">
            <h5>Epochs: {this.state.epochs}</h5>
          </div>
          <div className="col-6 col-lg-4">
            <h5>Guess: {this.state.guess}</h5>
          </div>
        </div>
        <div className="row justify-content-center pb-5">
          <div className="col">{probabilities}</div>
        </div>
      </div>
    );
  }
}

export default App;
