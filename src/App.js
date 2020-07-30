import React, { Component } from "react";
import Title from "./components/title";
import Description from "./components/description";
import Sketch from "./components/sketch";
import Button from "./components/button";
import FFNN from "./logic/ffnn";
import Matrix from "./logic/matrix";
import loadMNIST from "./logic/mnist";
import ffnnModel from "./models/ffnn-model.json";
import GithubCorner from "react-github-corner";

const OUTPUT_MNIST = true;
const INCLUDE_CROSS_VAL_SET = true;

const NUM_TRAINING = 5000;
const NUM_CROSS_VAL = 1000;
const NUM_TESTING = 1000;

const TRAIN_NETS = false;
const STANDARDIZATION_OVER_NORMALIZATION = true;
const EPSILON = 10 ** -100;

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      pixels: Array(784).fill(0),
      clearRequested: false,
      ffnnOutputArr: [],
      cnnOutputArr: [],
      ffnnPred: "",
      cnnPred: "",
    };

    if (TRAIN_NETS) {
      loadMNIST((data) => {
        this.mnist = data;
        console.log("All files loaded");

        if (OUTPUT_MNIST) {
          console.log(this.mnist);
        }

        // Only need to format if the current setting disagrees OR it's the first time formatting
        if (
          this.ffnn.useStandardization !== STANDARDIZATION_OVER_NORMALIZATION ||
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
        // this.ffnn.stochasticGradientDescent(
        //   this.trainingDatas.slice(0, this.state.trainingSize),
        //   1,
        //   10,
        //   0.03,
        //   1.0,
        //   this.testDatas
        // );

        // ffnn.chooseHypeparameters(
        //   this.trainingDatas,
        //   this.crossValDatas,
        //   this.testDatas
        // );
      });
    } else {
    }
    this.ffnn = new FFNN(null, ffnnModel);
  }

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
    if (!this.ffnn.hasOwnProperty("trainingMean")) {
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
      this.ffnn.trainingMean = mean;
      this.ffnn.trainingSTD = std;
    }

    // Retrieve mean and std from neural network
    const mean = this.ffnn.trainingMean;
    const std = this.ffnn.trainingSTD;

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

    this.ffnn.useStandardization = true;
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

    this.ffnn.useStandardization = false;
  };

  handleClickPredict = () => {
    if (this.ffnn !== undefined) {
      const input = Matrix.vectorFromArray(this.state.pixels);

      // Standardize or normalize input pixels
      if (this.ffnn.useStandardization) {
        input
          .sub(this.ffnn.trainingMean)
          .div(Matrix.map(this.ffnn.trainingSTD, (x) => x + EPSILON));
      } else {
        input.div(255);
      }

      const ffnnOutputArr = this.ffnn.feedforward(input);
      const ffnnPred = ffnnOutputArr.indexOf(Math.max(...ffnnOutputArr));

      // Update state about prediction and displayed probabilities
      this.setState({
        ffnnPred: ffnnPred,
        ffnnOutputArr,
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
      ffnnOutputArr: [],
      cnnOutputArr: [],
      ffnnPred: "",
      cnnPred: "",
    });
  };

  handleDraw = (pixels) => {
    this.setState({
      pixels,
    });
  };

  render() {
    const ffnnProbs = [];
    for (let i = 0; i < 10; i++) {
      const prob = (
        <div className="row" key={"fnn" + i}>
          <div className="col text-right">
            <h5>{i}:</h5>
          </div>
          <div className="col text-left">
            <h5>
              {this.state.ffnnOutputArr.length > 0
                ? (this.state.ffnnOutputArr[i] * 100).toFixed(1) + "%"
                : ""}
            </h5>
          </div>
        </div>
      );
      ffnnProbs.push(prob);
    }
    const cnnProbs = [];
    for (let i = 0; i < 10; i++) {
      const prob = (
        <div className="row" key={"cnn" + i}>
          <div className="col text-right">
            <h5>{i}:</h5>
          </div>
          <div className="col text-left">
            <h5>
              {this.state.cnnOutputArr.length > 0
                ? (this.state.cnnOutputArr[i] * 100).toFixed(1) + "%"
                : ""}
            </h5>
          </div>
        </div>
      );
      cnnProbs.push(prob);
    }

    return (
      <div className="App container text-center pt-5">
        <div className="row">
          <div className="col">
            <Title text="Digit Recognizer" />
          </div>
        </div>
        <div className="row">
          <div className="col">
            <Description
              text={
                "Digit recognition with feed forward (FFNN) and convolutional (CNN) neural networks."
              }
            />
          </div>
        </div>
        <div className="row justify-content-center pt-3">
          <div className="col-4 col-md-3 col-xl-2">
            <Button value="Predict" onClick={this.handleClickPredict} />
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
        <div className="row justify-content-center pb-5">
          <div className="col-6 col-md-4">
            <div className="row">
              <div className="col text-right">
                <h5>FFNN:</h5>
              </div>
              <div className="col text-left">
                <h5>{this.state.ffnnPred}</h5>
              </div>
            </div>
            {ffnnProbs}
          </div>
          <div className="col-6 col-md-4">
            <div className="row">
              <div className="col text-right">
                <h5>CNN:</h5>
              </div>
              <div className="col text-left">
                <h5>{this.state.cnnPred}</h5>
              </div>
            </div>
            {cnnProbs}
          </div>
        </div>
        <GithubCorner
          href="https://github.com/ryantran2165/handwritten-digits-neural-network"
          bannerColor="#222"
          octoColor="#7fffd4"
          target="_blank"
        />
      </div>
    );
  }
}

export default App;
