import React, { Component } from "react";
import p5 from "p5";
import PropTypes from "prop-types";

const IMG_SIZE = 28;
const BACKGROUND_COLOR = 211;

class Sketch extends Component {
  componentDidMount() {
    this.newSketch();
    window.addEventListener("resize", this.newSketch);
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.newSketch);
  }

  newSketch = () => {
    if (this.hasOwnProperty("myp5")) {
      this.myp5.remove();
    }

    this.myp5 = new p5((sketch) => {
      sketch.setup = () => {
        let width = Math.min(this.props.width, window.innerWidth * 0.8);
        let height = this.props.isSquare
          ? width
          : Math.min(this.props.height, window.innerHeight * 0.8);
        sketch.createCanvas(width, height);
        sketch.background("#d3d3d3");
      };

      sketch.draw = () => {
        if (this.props.clearRequested) {
          sketch.background("#d3d3d3");
          this.props.onClear();
        }
      };

      sketch.mouseDragged = () => {
        if (withinCanvas()) {
          sketch.strokeWeight(32);
          sketch.stroke(0);
          sketch.line(
            sketch.pmouseX,
            sketch.pmouseY,
            sketch.mouseX,
            sketch.mouseY
          );
          return false;
        }
      };

      sketch.mouseReleased = () => {
        if (withinCanvas()) {
          let img = sketch.get();
          img.resize(IMG_SIZE, IMG_SIZE);
          img.loadPixels();

          // Center drawing on center of mass
          let meanX = 0;
          let meanY = 0;
          let sumPixels = 0;
          const rows = img.height;
          const cols = img.width;
          for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
              const i = (y * cols + x) * 4;
              const pixel =
                img.pixels[i] === BACKGROUND_COLOR ? 0 : 255 - img.pixels[i];
              meanX += x * pixel;
              meanY += y * pixel;
              sumPixels += pixel;
            }
          }
          meanX /= sumPixels;
          meanY /= sumPixels;
          const dx = Math.round(cols / 2 - meanX);
          const dy = Math.round(rows / 2 - meanY);

          // Translate in x
          if (dx > 0) {
            // By row
            for (let y = 0; y < rows; y++) {
              // Start at the end, backwards until dx
              for (let x = cols - 1; x >= dx; x--) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = img.pixels[i - dx * 4];
              }

              // Fill beginning to dx with background color
              for (let x = 0; x < dx; x++) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = BACKGROUND_COLOR;
              }
            }
          } else if (dx < 0) {
            // By row
            for (let y = 0; y < rows; y++) {
              // Start at the beginning, forwards until end PLUS dx (dx is negative!)
              for (let x = 0; x < cols + dx; x++) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = img.pixels[i - dx * 4]; // dx is negative!
              }

              // Fill end plus dx (dx is negative!) to end with background color
              for (let x = cols + dx; x < cols; x++) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = BACKGROUND_COLOR;
              }
            }
          }

          // Translate in y
          if (dy > 0) {
            // By col
            for (let x = 0; x < cols; x++) {
              // Start at the end, backwards until dX
              for (let y = rows - 1; y >= dy; y--) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = img.pixels[i - dy * cols * 4];
              }

              // Fill beginning to dy with background color
              for (let y = 0; y < dy; y++) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = BACKGROUND_COLOR;
              }
            }
          } else if (dy < 0) {
            // By col
            for (let x = 0; x < cols; x++) {
              // Start at the beginning, forwards until end PLUS dy (dy is negative!)
              for (let y = 0; y < rows + dy; y++) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = img.pixels[i - dy * cols * 4];
              }

              // Fill end plus dy (dy is negative!) to end with background color
              for (let y = rows + dy; y < rows; y++) {
                const i = (y * cols + x) * 4;
                img.pixels[i] = BACKGROUND_COLOR;
              }
            }
          }

          // Extract only one channel of color
          let input = [];
          for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
            let color = img.pixels[i * 4];

            /*
              MNIST is black background = 0 with white text = 255.
              Sketch background is light gray = 211, so map it to black = 0.
              Sketch drawing is black = 0, so map it to white = 255.
              Normalization is delegated to the moment of prediction.
            */
            input[i] = color === BACKGROUND_COLOR ? 0 : 255 - color;
          }
          this.props.onDraw(input);

          return false;
        }
      };

      const withinCanvas = () => {
        return (
          sketch.mouseX > 0 &&
          sketch.mouseX < sketch.width &&
          sketch.mouseY > 0 &&
          sketch.mouseY < sketch.height
        );
      };
    }, "p5sketch");
  };

  render() {
    return <div id="p5sketch"></div>;
  }
}

Sketch.defaultProps = {
  width: 100,
  height: 100,
  isSquare: true,
};

Sketch.propTypes = {
  width: PropTypes.number,
  height: PropTypes.number,
  isSquare: PropTypes.bool,
  onDraw: PropTypes.func,
  clearRequested: PropTypes.bool,
  onClear: PropTypes.func,
};

export default Sketch;
