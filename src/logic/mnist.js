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

  // Load all files
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
  // Fetch file response
  let response = await fetch(file);

  // Get 8-bit/1-byte array
  let buffer = await response.arrayBuffer();

  // Default header count is 4 for images, change to 2 for labels later
  let headerCount = 4;

  // Can't access data straight from ArrayBuffer, extract headers using DataView
  let headerView = new DataView(buffer, 0, 4 * headerCount);

  // Create Array of 32-bit/4-byte headers
  let headers = new Array(headerCount)
    .fill()
    .map((_, i) => headerView.getUint32(4 * i, false));

  // Get file type: image or label
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

  // Create array of data only, headers removed
  let data = new Uint8Array(buffer, headerCount * 4);

  // Image, create array of subarrays of 28 * 28 = 784
  if (type === "image") {
    let dataArr = [];
    for (let i = 0; i < headers[1]; i++) {
      dataArr.push(data.subarray(dataLength * i, dataLength * (i + 1)));
    }
    console.log("Loaded file:", file);
    return dataArr;
  }

  // Label, just return data straight away
  console.log("Loaded file:", file);
  return data;
}

export default loadMNIST;
