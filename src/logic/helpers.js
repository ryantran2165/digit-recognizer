import Matrix from "./matrix";

export const sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x));
};

export const sigmoidDerivative = (x) => {
  return sigmoid(x) * (1 - sigmoid(x));
};

export class QuadraticCost {
  static fn = (a, y) => {
    return 0;
  };

  static delta = (z, a, y) => {
    return Matrix.sub(a, y).mul(Matrix.map(z, sigmoidDerivative));
  };
}

export class CrossEntropyCost {
  static fn = (a, y) => {
    return 0;
  };

  static delta = (z, a, y) => {
    return Matrix.sub(a, y);
  };
}

/**
 * Fisher-Yates shuffle.
 * @param {Array} arr The array to shuffle
 * @return {Array} The shuffled array
 */
export const shuffle = (arr) => {
  for (let i = arr.length - 1; i > 0; i--) {
    const randomIndex = Math.floor(Math.random() * (i + 1));
    const temp = arr[i];
    arr[i] = arr[randomIndex];
    arr[randomIndex] = temp;
  }
  return arr;
};
