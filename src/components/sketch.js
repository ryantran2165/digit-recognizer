import React, { Component } from "react";
import p5 from "p5";
import PropTypes from "prop-types";

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
          img.resize(28, 28);
          img.loadPixels();

          let input = [];
          for (let i = 0; i < 28 * 28; i++) {
            let color = img.pixels[i * 4];
            input[i] = color === 211 ? 0 : (255 - color) / 255;
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
