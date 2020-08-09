import Matrix from "../matrix";

class Conv {
  /**
   * Create a Conv layer with the given settings or loads a pretrained one.
   * @param {number} numFilters The number of filters
   * @param {number} filterSize The filter size, square only
   * @param {Conv} conv The pretrained Conv
   */
  constructor(numFilters, filterSize = 3, conv = null) {
    if (conv !== null) {
      this.numFilters = conv.numFilters;
      this.filterSize = conv.filterSize;
      this.filters = [];

      for (let filter of conv.filters) {
        this.filters.push(new Matrix(null, null, filter));
      }
    } else {
      this.numFilters = numFilters;
      this.filterSize = filterSize;
      this.filters = [];

      for (let i = 0; i < numFilters; i++) {
        this.filters.push(
          new Matrix(filterSize, filterSize)
            .randomizeNormal()
            .div(filterSize ** 2)
        );
      }
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
    const f = this.filterSize;

    for (let y = 0; y < h - f + 1; y++) {
      for (let x = 0; x < w - f + 1; x++) {
        yield [image.getRegion(x, y, f, f), y, x];
      }
    }
  }

  /**
   * Feedforward implementation assuming a very simple CNN architecture (single channel).
   * @param {Matrix} input The input image
   * @return {Array} An array of images processed by the filters
   */
  forward(input) {
    this.lastInput = input;

    const h = input.rows;
    const w = input.cols;
    const f = this.filterSize;

    const outputs = [];

    // For each filter
    for (let filter of this.filters) {
      const output = new Matrix(h - f + 1, w - f + 1);

      // For each region
      for (let [region, y, x] of this.iterateRegions(input)) {
        // Apply the filter to the region
        for (let r = 0; r < f; r++) {
          for (let c = 0; c < f; c++) {
            output.data[y][x] += region.data[r][c] * filter.data[r][c];
          }
        }
      }

      outputs.push(output);
    }

    return outputs;
  }

  /**
   * Backpropagation assuming only a single channel.
   * @param {Array} dLdOut Array of loss gradients w.r.t. the conv layer's outputs
   * @param {number} learningRate The learning rate
   */
  backprop(dLdOut, learningRate) {
    const dLdFilters = [];

    // For each filter
    for (let i = 0; i < this.filters.length; i++) {
      const filter = this.filters[i];
      const dLdFilter = new Matrix(filter.rows, filter.cols);

      // For each region
      for (let [region, y, x] of this.iterateRegions(this.lastInput)) {
        // Accumulate filter gradient
        dLdFilter.add(region.mul(dLdOut[i].data[y][x]));
      }
      dLdFilters.push(dLdFilter);
    }

    // Update filters
    for (let i = 0; i < this.filters.length; i++) {
      const filter = this.filters[i];
      const dLdFilter = dLdFilters[i];

      filter.sub(dLdFilter.mul(learningRate));
    }
  }
}

export default Conv;
