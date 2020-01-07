import React, { Component } from "react";
import Title from "./components/title";
import Description from "./components/description";
import Sketch from "./components/sketch";
import Button from "./components/button";
import NeuralNetwork from "./logic/neural-network";
import Matrix from "./logic/matrix";
import loadMNIST from "./logic/mnist";
import neuralNetworkPretrained from "./neural-network-pretrained.json";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      epochs: 0,
      guess: "",
      pixels: Array(784).fill(0),
      clearRequested: false,
      isTraining: false,
      usePretrained: true
    };

    this.dataFormatted = false;
    loadMNIST(data => {
      this.mnist = data;
      this.neuralNetwork = new NeuralNetwork(null, neuralNetworkPretrained);
    });
  }

  handleClickTrain = () => {
    if (this.neuralNetwork !== undefined) {
      this.setState({ isTraining: true });
      this.timer = setTimeout(() => {
        if (!this.dataFormatted) {
          this.dataFormatted = true;

          const loadDatas = (images, labels) => {
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
                Matrix.fromArray(inputArr).map(x => x / 255),
                Matrix.fromArray(desiredArr)
              ]);
            }

            return datas;
          };

          this.trainingDatas = loadDatas(
            this.mnist.trainingImages.slice(0, 50000),
            this.mnist.trainingLabels.slice(0, 50000)
          );
          // this.testDatas = loadDatas(
          //   this.mnist.testImages.slice(0, 10000),
          //   this.mnist.testLabels.slice(0, 10000)
          // );
        }

        this.neuralNetwork.stochasticGradientDescent(
          this.trainingDatas,
          1,
          10,
          3.0
          // this.testDatas
        );

        this.setState(prevState => ({
          epochs: prevState.epochs + 1,
          isTraining: false
        }));

        clearTimeout(this.timer);
      }, 250);
    }
  };

  handleClickGuess = () => {
    if (this.neuralNetwork !== undefined) {
      const input = Matrix.fromArray(this.state.pixels);
      const outputArr = this.neuralNetwork.feedforward(input);
      const guess = outputArr.indexOf(Math.max(...outputArr));
      this.setState({
        guess
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
      guess: ""
    });
  };

  handleDraw = pixels => {
    this.setState({
      pixels
    });
  };

  handleSwitchChange = () => {
    this.setState(
      prevState => ({ usePretrained: !prevState.usePretrained }),
      () => {
        this.neuralNetwork = this.state.usePretrained
          ? new NeuralNetwork(null, neuralNetworkPretrained)
          : new NeuralNetwork([784, 30, 10]);
        this.setState({ epochs: 0 });
      }
    );
  };

  render() {
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
                'Recognizes handwritten digits using a neural network.\nDraw a number from 0 to 9 and have the neural network guess what you drew.\nTry to center and scale your drawing to fill the majority of the canvas.\nPress the "train" button to train the neural network for one epoch.\nOr just use the pre-trained neural network.\n*Training may take a few minutes depending on your computer.'
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
              onDraw={pixels => this.handleDraw(pixels)}
              clearRequested={this.state.clearRequested}
              onClear={this.handleClear}
            />
          </div>
        </div>
        <div className="row justify-content-center pb-5">
          <div className="col-6 col-lg-4">
            <h5>Epochs: {this.state.epochs}</h5>
          </div>
          <div className="col-6 col-lg-4">
            <h5>Guess: {this.state.guess}</h5>
          </div>
        </div>
      </div>
    );
  }
}

export default App;
