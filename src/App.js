import React, { Component } from "react";
import Title from "./components/title";
import Description from "./components/description";
import Button from "./components/button";
import FFNN from "./logic/ffnn";
import Matrix from "./logic/matrix";
import loadMNIST from "./logic/mnist";
import ffnnModel from "./models/ffnn-model.json";
import GithubCorner from "react-github-corner";
import Canvas from "./components/canvas";

const TRAIN_NET = false;
const NEW_NET = false;
const USE_STANDARDIZATION = false;

const LOAD_TRAIN = false;
const LOAD_VAL = false;
const LOAD_TEST = true;

const OUTPUT_ACCURACY = false;
const OUTPUT_MNIST = false;

const NUM_TRAIN = 50000;
const NUM_VAL = 10000;
const NUM_TEST = 150;
const EPSILON = 10 ** -100;

const NUM_TEST_SAMPLES = 5;

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      ffnnOutputArr: [],
      cnnOutputArr: [],
      ffnnPred: "",
      cnnPred: "",
    };

    loadMNIST((data) => {
      this.mnist = data;
      console.log("All files loaded");

      if (OUTPUT_MNIST) {
        console.log(this.mnist);
      }

      this.formatData();
      if (USE_STANDARDIZATION) {
        this.standardizeData();
      } else {
        this.normalizeData();
      }

      this.showSamples();

      if (TRAIN_NET) {
        console.log("Starting training");

        if (NEW_NET) {
          this.ffnn = new FFNN([784, 30, 10]);
        } else {
          this.ffnn = new FFNN(null, ffnnModel);
        }

        // Remember to use much smaller learning rate when using ReLU
        this.ffnn.stochasticGradientDescent(
          this.trainDatas,
          1,
          10,
          0.03,
          1.0,
          this.testDatas
        );
      } else {
        this.ffnn = new FFNN(null, ffnnModel);

        if (OUTPUT_ACCURACY) {
          const accuracy = this.ffnn.accuracy(this.testDatas);
          console.log(
            "Accuracy: " +
              accuracy +
              "/" +
              this.testDatas.length +
              ", " +
              (100 * accuracy) / this.testDatas.length +
              "%"
          );
        }
      }
    });

    this.canvasRef = React.createRef();
  }

  showSamples = () => {
    // Find test images for digit i
    for (let i = 0; i < 10; i++) {
      let testImages = [];

      for (let testData of this.testDatas) {
        const labelArr = testData[1].toArray();
        const label = labelArr.indexOf(Math.max(...labelArr));

        // Found digit
        if (label === i) {
          testImages.push(testData[0].toArray());

          // Found required number of digits
          if (testImages.length === NUM_TEST_SAMPLES) {
            break;
          }
        }
      }

      // Display sample test images
      for (let j = 0; j < 5; j++) {
        const canvas = document.getElementById(`canvas-${i}-${j}`);
        const ctx = canvas.getContext("2d");

        for (let y = 0; y < 28; y++) {
          for (let x = 0; x < 28; x++) {
            const block = ctx.createImageData(1, 1);
            const newVal = 255 * testImages[j][y * 28 + x];

            block.data[0] = newVal;
            block.data[1] = newVal;
            block.data[2] = newVal;
            block.data[3] = 255;

            ctx.putImageData(block, x, y);
          }
        }
      }
    }
  };

  formatData = () => {
    if (LOAD_TRAIN) {
      console.log("Formatting train data");
      this.trainDatas = this.loadDatas(
        this.mnist.trainImages.slice(0, NUM_TRAIN),
        this.mnist.trainLabels.slice(0, NUM_TRAIN)
      );
    }

    if (LOAD_VAL) {
      console.log("Formatting validation data");
      this.valDatas = this.loadDatas(
        this.mnist.trainImages.slice(NUM_TRAIN, NUM_TRAIN + NUM_VAL),
        this.mnist.trainLabels.slice(NUM_TRAIN, NUM_TRAIN + NUM_VAL)
      );
    }

    if (LOAD_TEST) {
      console.log("Formatting test data");
      this.testDatas = this.loadDatas(
        this.mnist.testImages.slice(0, NUM_TEST),
        this.mnist.testLabels.slice(0, NUM_TEST)
      );
    }
  };

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
    // Neural network does NOT have predefined train mean and STD, calculate them and use that to standardize everything
    if (!this.ffnn.hasOwnProperty("trainMean")) {
      const mean = new Matrix(784, 1); // mean = sum(pixels) / numTrain
      const std = new Matrix(784, 1); // std = sqrt(sum((pixels - mean)^2) / numTrain)

      // Calculate sum(pixels)
      console.log("Calculating train data mean");
      for (let trainData of this.trainDatas) {
        mean.add(trainData[0]);
      }

      // Divide by total number of train datas
      mean.map((x) => x / this.trainDatas.length);

      // Calculate sum((pixels - mean)^2)
      console.log("Calculating train data standard deviation");
      for (let trainData of this.trainDatas) {
        std.add(Matrix.sub(trainData[0], mean).map((x) => x ** 2));
      }

      // Divide by total number of train datas and square root
      std.map((x) => Math.sqrt(x / this.trainDatas.length));

      // Save to neural network
      this.ffnn.trainMean = mean;
      this.ffnn.trainSTD = std;
    }

    // Retrieve mean and std from neural network
    const mean = this.ffnn.trainMean;
    const std = this.ffnn.trainSTD;

    // Standardize train images, add small epsilon so never divide by zero
    if (LOAD_TRAIN) {
      console.log("Standardizing train data");
      for (let i = 0; i < this.trainDatas.length; i++) {
        this.trainDatas[i][0]
          .sub(mean)
          .div(Matrix.map(std, (x) => x + EPSILON));
      }
    }

    if (LOAD_VAL) {
      // Standardize validation images, add small epsilon so never divide by zero
      console.log("Standardizing validation data");
      for (let i = 0; i < this.valDatas.length; i++) {
        this.valDatas[i][0].sub(mean).div(Matrix.map(std, (x) => x + EPSILON));
      }
    }

    if (LOAD_TEST) {
      // Standardize test images, add small epsilon so never divide by zero
      console.log("Standardizing test data");
      for (let i = 0; i < this.testDatas.length; i++) {
        this.testDatas[i][0].sub(mean).div(Matrix.map(std, (x) => x + EPSILON));
      }
    }
  };

  normalizeData = () => {
    // Normalize train images
    if (LOAD_TRAIN) {
      for (let trainData of this.trainDatas) {
        trainData[0].div(255);
      }
    }

    if (LOAD_VAL) {
      // Normalize validation images
      for (let valData of this.valDatas) {
        valData[0].div(255);
      }
    }

    if (LOAD_TEST) {
      // Normalize test images
      for (let testData of this.testDatas) {
        testData[0].div(255);
      }
    }
  };

  handlePredict = () => {
    if (this.ffnn !== undefined) {
      const input = Matrix.vectorFromArray(this.canvasRef.current.predict());
      // input.mul(255);
      // input.map(Math.round);

      // Standardize or normalize input pixels
      if (USE_STANDARDIZATION) {
        input
          .sub(this.ffnn.trainMean)
          .div(Matrix.map(this.ffnn.trainSTD, (x) => x + EPSILON));
      } else {
        // input.div(255);
      }
      const ffnnOutputArr = this.ffnn.feedforward(input);
      const ffnnPred = ffnnOutputArr.indexOf(Math.max(...ffnnOutputArr));

      // Update state about prediction and displayed probabilities
      this.setState({
        ffnnOutputArr,
        ffnnPred,
      });
    }
  };

  handleClear = () => {
    this.canvasRef.current.erase();

    this.setState({
      ffnnOutputArr: [],
      cnnOutputArr: [],
      ffnnPred: "",
      cnnPred: "",
    });
  };

  render() {
    const testImages = [];
    for (let i = 0; i < 10; i++) {
      const cols = [];

      for (let j = 0; j < NUM_TEST_SAMPLES; j++) {
        const col = (
          <div className="col-auto" key={`col-${i}-${j}`}>
            <canvas
              id={`canvas-${i}-${j}`}
              width="28"
              height="28"
              style={{
                border: "2px solid aquamarine",
                borderRadius: "5px",
                backgroundColor: "white",
              }}
            ></canvas>
          </div>
        );
        cols.push(col);
      }

      const row = (
        <div className="row justify-content-center" key={`row-${i}`}>
          {cols}
        </div>
      );
      testImages.push(row);
    }

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
        <div className="row pt-3">
          <div className="col">
            <h3>Test Samples:</h3>
          </div>
        </div>
        {testImages}
        <div className="row justify-content-center pt-5">
          <div className="col-4 col-md-3 col-xl-2">
            <Button value="Predict" onClick={this.handlePredict} />
          </div>
          <div className="col-4 col-md-3 col-xl-2">
            <Button value="Clear" onClick={this.handleClear} />
          </div>
        </div>
        <div className="row pt-3">
          <div className="col">
            <Canvas ref={this.canvasRef} />
          </div>
        </div>
        <div className="row">
          <div className="col">
            <canvas
              id="canvas28"
              width="28"
              height="28"
              style={{
                border: "2px solid aquamarine",
                borderRadius: "5px",
                backgroundColor: "white",
              }}
            ></canvas>
          </div>
        </div>
        <div className="row justify-content-center pb-5">
          <div className="col-6 col-md-4 col-lg-3">
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
          <div className="col-6 col-md-4 col-lg-3">
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
          href="https://github.com/ryantran2165/digit-recognizer"
          bannerColor="#222"
          octoColor="#7fffd4"
          target="_blank"
        />
      </div>
    );
  }
}

export default App;
