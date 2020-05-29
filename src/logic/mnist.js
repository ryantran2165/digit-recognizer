/**
 * Loads the MNIST data.
 * @param {function} callback The callback function to be called when loading is finished
 * @return {Promise} The resolved promise
 */
function loadMNIST(callback) {
  let mnist = {};
  let files = {
    trainingImages: "./train-images.idx3-ubyte",
    trainingLabels: "./train-labels.idx1-ubyte",
    testImages: "./t10k-images.idx3-ubyte",
    testLabels: "./t10k-labels.idx1-ubyte",
  };
  return Promise.all(
    Object.keys(files).map(async (file) => {
      mnist[file] = await loadFile(files[file]);
    })
  ).then(() => callback(mnist));
}

/**
 * Parses the MNIST file into an array of data.
 * @param {string} file The filename
 * @return {Array} The MNIST data
 */
async function loadFile(file) {
  let response = await fetch(file);
  let buffer = await response.arrayBuffer();
  let headerCount = 4;
  let headerView = new DataView(buffer, 0, 4 * headerCount);
  let headers = new Array(headerCount)
    .fill()
    .map((_, i) => headerView.getUint32(4 * i, false));

  let type, dataLength;
  if (headers[0] === 2049) {
    type = "label";
    dataLength = 1;
    headerCount = 2;
  } else if (headers[0] === 2051) {
    type = "image";
    dataLength = headers[2] * headers[3];
  } else {
    throw new Error("Unknown file type " + headers[0]);
  }

  let data = new Uint8Array(buffer, headerCount * 4);
  if (type === "image") {
    let dataArr = [];
    for (let i = 0; i < headers[1]; i++) {
      dataArr.push(data.subarray(dataLength * i, dataLength * (i + 1)));
    }
    console.log("Loaded file:", file);
    return dataArr;
  }
  console.log("Loaded file:", file);
  return data;
}

export default loadMNIST;
