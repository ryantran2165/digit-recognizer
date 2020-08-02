import React, { Component } from "react";

const WIDTH = 280;
const HEIGHT = 280;
const SCALE_SIZE = 190;
const BOUNDING_THRESHOLD = 0.01;
const LINE_WIDTH = 20;
const COLOR = "black";
const SCALE_PATH = true;

const DEBUG = false;

class Canvas extends Component {
  componentDidMount() {
    this.canvas = document.getElementById("main-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.canvas28 = document.getElementById("canvas28");
    this.ctx28 = this.canvas28.getContext("2d");

    this.canvas.addEventListener(
      "mousemove",
      (e) => {
        this.findxy("move", e);
      },
      false
    );
    this.canvas.addEventListener(
      "mousedown",
      (e) => {
        this.findxy("down", e);
      },
      false
    );
    this.canvas.addEventListener(
      "mouseup",
      (e) => {
        this.findxy("up", e);
      },
      false
    );
    this.canvas.addEventListener(
      "mouseout",
      (e) => {
        this.findxy("out", e);
      },
      false
    );

    this.prevX = 0;
    this.currX = 0;
    this.prevY = 0;
    this.currY = 0;
    this.paths = [];
    this.paintFlag = false;
  }

  findxy = (res, e) => {
    if (res === "down") {
      // Get touch down point
      this.currX = e.clientX - this.canvas.getBoundingClientRect().left;
      this.currY = e.clientY - this.canvas.getBoundingClientRect().top;

      // Draw circle
      this.ctx.beginPath();
      this.ctx.lineWidth = 1;
      this.ctx.arc(this.currX, this.currY, LINE_WIDTH / 2, 0, 2 * Math.PI);
      this.ctx.stroke();
      this.ctx.closePath();
      this.ctx.fill();

      // Add start of new path
      this.paths.push([[this.currX], [this.currY]]);

      // Activate paint flag
      this.paintFlag = true;
    }

    // Deactivate paint flag on touch up or mouse off canvas
    if (res === "up" || res === "out") {
      this.paintFlag = false;
    }

    // If moving and paint flag activated, draw!
    if (res === "move" && this.paintFlag) {
      // Save previous point
      this.prevX = this.currX;
      this.prevY = this.currY;

      // Get new point
      this.currX = e.clientX - this.canvas.getBoundingClientRect().left;
      this.currY = e.clientY - this.canvas.getBoundingClientRect().top;

      // Add point to last path
      const currPath = this.paths[this.paths.length - 1];
      currPath[0].push(this.currX);
      currPath[1].push(this.currY);
      this.paths[this.paths.length - 1] = currPath;

      // Draw line between new point and previous point
      this.draw(
        this.ctx,
        LINE_WIDTH,
        this.prevX,
        this.prevY,
        this.currX,
        this.currY
      );
    }
  };

  // Draw line between 2 points
  draw = (ctx, lineWidth, x1, y1, x2, y2) => {
    ctx.beginPath();
    ctx.strokeStyle = COLOR;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.closePath();
  };

  // Erase canvas and clear paths
  erase = () => {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.paths = [];

    this.ctx28.clearRect(0, 0, this.canvas28.width, this.canvas28.height);
  };

  imageDataToGrayscale = (imgData) => {
    // 2D array
    const grayscaleImg = [];

    for (let y = 0; y < imgData.height; y++) {
      // Row number y
      grayscaleImg[y] = [];

      for (let x = 0; x < imgData.width; x++) {
        // Each pixel has four values, RGBA
        const offset = y * 4 * imgData.width + 4 * x;

        // Alpha === 0 means no drawing
        const alpha = imgData.data[offset + 3];

        // No drawing, set value to white
        if (alpha === 0) {
          imgData.data[offset] = 255;
        }

        // Take only RED value since grayscale and scale to [0, 1]
        grayscaleImg[y][x] = imgData.data[offset] / 255;
        // grayscaleImg[y][x] = imgData.data[offset];
      }
    }

    return grayscaleImg;
  };

  getBoundingRect = (img, threshold) => {
    const rows = img.length;
    const columns = img[0].length;
    let minX = columns;
    let minY = rows;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        // Black === 0, so must be lower than some darkness threshold to be considered significant
        if (img[y][x] < threshold) {
          if (minX > x) minX = x;
          if (maxX < x) maxX = x;
          if (minY > y) minY = y;
          if (maxY < y) maxY = y;
        }
      }
    }

    return { minY: minY, minX: minX, maxY: maxY, maxX: maxX };
  };

  centerImage = (img) => {
    const rows = img.length;
    const columns = img[0].length;
    let meanX = 0;
    let meanY = 0;
    let sumPixels = 0;

    // Center of mass, weighted by color intensity
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        let pixel = 1 - img[y][x];
        meanX += x * pixel;
        meanY += y * pixel;
        sumPixels += pixel;
      }
    }
    meanX /= sumPixels;
    meanY /= sumPixels;

    let dx = Math.round(columns / 2 - meanX);
    let dy = Math.round(rows / 2 - meanY);

    return { transX: dx, transY: dy };
  };

  predict = () => {
    // Get image and convert to grayscale
    let imgData = this.ctx.getImageData(0, 0, WIDTH, HEIGHT);
    let grayscaleImg = this.imageDataToGrayscale(imgData);

    // Get bounding rectangle and center of mass translation amount
    const boundingRect = this.getBoundingRect(grayscaleImg, BOUNDING_THRESHOLD);
    // const boundingRect = this.getBoundingRect(grayscaleImg, 25);
    const trans = this.centerImage(grayscaleImg);

    // Create hidden copy of canvas context
    const canvasCopy = document.createElement("canvas");
    canvasCopy.width = imgData.width;
    canvasCopy.height = imgData.height;
    const copyCtx = canvasCopy.getContext("2d");

    // Scale largest dimension to SCALE_SIZE
    const brW = boundingRect.maxX - boundingRect.minX + 1;
    const brH = boundingRect.maxY - boundingRect.minY + 1;
    const scaling = SCALE_SIZE / (brW > brH ? brW : brH);

    // Scale
    copyCtx.translate(this.canvas.width / 2, this.canvas.height / 2);
    copyCtx.scale(scaling, scaling);
    copyCtx.translate(-this.canvas.width / 2, -this.canvas.height / 2);

    // Center on canvas over center of mass
    copyCtx.translate(trans.transX, trans.transY);

    if (SCALE_PATH) {
      // Scale path line width
      for (let p = 0; p < this.paths.length; p++) {
        for (let i = 0; i < this.paths[p][0].length - 1; i++) {
          const x1 = this.paths[p][0][i];
          const y1 = this.paths[p][1][i];
          const x2 = this.paths[p][0][i + 1];
          const y2 = this.paths[p][1][i + 1];
          this.draw(copyCtx, LINE_WIDTH / scaling, x1, y1, x2, y2);
        }
      }
    } else {
      copyCtx.drawImage(this.ctx.canvas, 0, 0);
    }

    // Get scaled and translated image and convert to grayscale
    imgData = copyCtx.getImageData(0, 0, WIDTH, HEIGHT);
    grayscaleImg = this.imageDataToGrayscale(imgData);

    // Final input array for neural net
    const nnInput = new Array(784);

    // Convert to 28x28
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        let mean = 0;
        for (let v = 0; v < 10; v++) {
          for (let h = 0; h < 10; h++) {
            mean += grayscaleImg[y * 10 + v][x * 10 + h];
          }
        }

        // Average and invert color
        nnInput[y * 28 + x] = 1 - mean / 100;
        // nnInput[y + 28 * x] = 255 - mean / 100;
      }
    }

    // Draw to 28x28 canvas
    this.ctx28.clearRect(0, 0, this.canvas28.width, this.canvas28.height);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const block = this.ctx28.getImageData(x, y, 1, 1);
        const newVal = 255 * nnInput[y * 28 + x];

        block.data[0] = newVal;
        block.data[1] = newVal;
        block.data[2] = newVal;
        block.data[3] = 255;

        this.ctx28.putImageData(block, x, y);
      }
    }

    // Draw neural network input back to canvas
    if (DEBUG) {
      // Clear canvas
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          // Blocks of 10
          const block = this.ctx.getImageData(x * 10, y * 10, 10, 10);
          const newVal = 255 * nnInput[y * 28 + x];
          // const newVal = nnInput[y + 28 * x];

          // R=G=B since grayscale, A=255 for full opacity
          for (let i = 0; i < 4 * 10 * 10; i += 4) {
            block.data[i] = newVal;
            block.data[i + 1] = newVal;
            block.data[i + 2] = newVal;
            block.data[i + 3] = 255;
          }

          // Paint new data to canvas
          this.ctx.putImageData(block, x * 10, y * 10);
        }
      }
    }

    return nnInput;
  };

  render() {
    return (
      <canvas
        id="main-canvas"
        width={WIDTH.toString()}
        height={HEIGHT.toString()}
        style={{
          border: "5px solid aquamarine",
          borderRadius: "5px",
          backgroundColor: "white",
        }}
      ></canvas>
    );
  }
}

export default Canvas;
