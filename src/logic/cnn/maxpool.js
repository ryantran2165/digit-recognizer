import Matrix from "../matrix";

class MaxPool {
  /**
   * Creates a max pool with the given size or loads the pool size from the given pool.
   * @param {number} poolSize The pool size, square only, stride same as size
   * @param {MaxPool} pool The preset pool, only containing size information
   */
  constructor(poolSize = 2, pool = null) {
    if (pool !== null) {
      this.poolSize = pool.poolSize;
    } else {
      this.poolSize = poolSize;
    }
  }

  /**
   * Generator method for iterating over the image.
   * @param {Matrix} image The image to iterate over
   * @return {Array} The image region as a 2D Matrix and the coordinates of the region
   */
  *iterateRegions(image) {
    const h = image.rows;
    const w = image.cols;
    const s = this.poolSize;

    const newH = Math.floor(h / s);
    const newW = Math.floor(w / s);

    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        yield [image.getRegion(x * s, y * s, s, s), y, x];
      }
    }
  }

  /**
   * Feedforward on the max pool.
   * @param {Array} input The array of input images passed through a conv layer
   * @return {Array} An array of images processed by the max pool
   */
  forward(input) {
    this.lastInput = input;

    const h = input[0].rows;
    const w = input[0].cols;
    const s = this.poolSize;

    const newH = Math.floor(h / s);
    const newW = Math.floor(w / s);

    const outputs = [];

    // For each channel
    for (let channel of input) {
      const output = new Matrix(newH, newW);

      // For each region
      for (let [region, y, x] of this.iterateRegions(channel)) {
        // Apply the pool to the region
        const [max, _, __] = region.max();
        output.data[y][x] = max;
      }

      outputs.push(output);
    }

    return outputs;
  }

  /**
   * Backpropagation on the max pool.
   * @param {Array} dLdOut Array of loss gradients w.r.t. the pool layer's outputs
   * @return {Array} Array of loss gradients w.r.t. the pool layer's inputs
   */
  backprop(dLdOut) {
    const dLdInputs = [];

    // For each channel
    for (let i = 0; i < this.lastInput.length; i++) {
      const channel = this.lastInput[i];
      const dLdInput = new Matrix(channel.rows, channel.cols);

      // For each region
      for (let [region, y, x] of this.iterateRegions(channel)) {
        // Apply the pool to the region
        const [max, _, __] = region.max();

        // Find all pixels matching the max value and replace with corresponding dLdOut
        for (let r = 0; r < region.rows; r++) {
          for (let c = 0; c < region.cols; c++) {
            if (region.data[r][c] === max) {
              dLdInput.data[y * this.poolSize + r][x * this.poolSize + c] =
                dLdOut[i].data[y][x];
            }
          }
        }
      }

      dLdInputs.push(dLdInput);
    }

    return dLdInputs;
  }
}

export default MaxPool;
