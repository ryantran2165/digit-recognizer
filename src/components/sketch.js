import React, { Component } from "react";
import p5 from "p5";
import PropTypes from "prop-types";

class Sketch extends Component {
  constructor(props) {
    super(props);
    this.renderRef = new React.createRef();
    this.newSketch();
  }

  componentDidMount() {
    window.addEventListener("resize", this.newSketch);
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.newSketch);
  }

  newSketch = () => {
    if (this.hasOwnProperty("sketch")) {
      this.sketch.remove();
    }

    this.sketch = new p5(p => {
      p.setup = () => {
        let width = Math.min(this.props.width, window.innerWidth * 0.8);
        let height = this.props.isSquare
          ? width
          : Math.min(this.props.height, window.innerHeight * 0.8);
        p.createCanvas(width, height).parent(this.renderRef.current);
        p.background("#d3d3d3");
      };

      p.draw = () => {
        if (this.props.clearRequested) {
          p.background("#d3d3d3");
          this.props.onClear();
        }

        p.strokeWeight(32);
        p.stroke(0);
        if (
          p.mouseIsPressed &&
          p.mouseX > 0 &&
          p.mouseX < p.width &&
          p.mouseY > 0 &&
          p.mouseY < p.height
        ) {
          p.line(p.pmouseX, p.pmouseY, p.mouseX, p.mouseY);
        }
      };

      p.mouseReleased = () => {
        if (
          p.mouseX > 0 &&
          p.mouseX < p.width &&
          p.mouseY > 0 &&
          p.mouseY < p.height
        ) {
          let img = p.get();
          img.resize(28, 28);
          img.loadPixels();

          let input = [];
          for (let i = 0; i < 28 * 28; i++) {
            let color = img.pixels[i * 4];
            input[i] = color === 211 ? 0 : (255 - color) / 255;
          }
          this.props.onDraw(input);
        }
        return false;
      };
    });
  };

  render() {
    return <div ref={this.renderRef}></div>;
  }
}

Sketch.defaultProps = {
  width: 100,
  height: 100,
  isSquare: true
};

Sketch.propTypes = {
  width: PropTypes.number,
  height: PropTypes.number,
  isSquare: PropTypes.bool,
  onDraw: PropTypes.func,
  clearRequested: PropTypes.bool,
  onClear: PropTypes.func
};

export default Sketch;
