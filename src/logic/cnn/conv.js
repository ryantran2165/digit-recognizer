import Matrix from "../matrix";

class Conv {
  constructor(numFilters, filterSize = 3) {
    this.numFilters = numFilters;
    this.filters = [];
    this.filterSize = filterSize;

    for (let i = 0; i < numFilters; i++) {
      this.filters.push(
        new Matrix(filterSize, filterSize)
          .randomizeNormal()
          .div(filterSize ** 2)
      );
    }
  }

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

        // for (let r = 0; r < region.rows; r++) {
        //   for (let c = 0; c < region.cols; c++) {
        //     dLdFilter.data[r][c] += dLdOut[i].data[y][x] * region.data[r][c];
        //   }
        // }
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
