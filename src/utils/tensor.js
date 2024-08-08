import * as tf from "@tensorflow/tfjs";

/**
 * Scale and position the bounding box for the original frame before cropping and resizing
 * @param {Array} box The bounding box [y1, x1, y2, x2] in the model's output coordinates
 * @param {number} xRatio The ratio used for scaling the x coordinates
 * @param {number} yRatio The ratio used for scaling the y coordinates
 * @param {Array} faceBox The bounding box [x1, y1, x2, y2] used for cropping the face
 * @param {number} originalWidth The original width of the image/frame
 * @param {number} originalHeight The original height of the image/frame
 * @returns {Array} Scaled bounding box [y1, x1, y2, x2] in the original image coordinates
 */
export const scaleAndPositionBoundingBox = (box, xRatio, yRatio, faceBox, originalWidth, originalHeight) => {
    // Extract coordinates
    let [y1, x1, y2, x2] = box;
  
    // Apply scaling ratios
    x1 = x1 / xRatio;
    y1 = y1 / yRatio;
    x2 = x2 / xRatio;
    y2 = y2 / yRatio;
  
    // Adjust coordinates if a faceBox was used for cropping
    if (faceBox) {
      const [faceX1, faceY1] = faceBox;
      x1 += faceX1;
      y1 += faceY1;
      x2 += faceX1;
      y2 += faceY1;
    }
  
    return [Math.round(y1), Math.round(x1), Math.round(y2), Math.round(x2)];
  };

/**
 * Create a bounding box given center point and dimension
 * @param {number} centerX The x-coordinate of the center point
 * @param {number} centerY The y-coordinate of the center point
 * @param {number} dimension The width and height of the bounding box (square)
 * @returns {Array} Bounding box [x1, y1, x2, y2]
 */
export const createBoundingBoxFromCenter = (centerX, centerY, dimension) => {
    const halfDim = dimension / 2;
  
    // Calculate the coordinates of the bounding box
    const x1 = centerX - halfDim;
    const y1 = centerY - halfDim;
    const x2 = centerX + halfDim;
    const y2 = centerY + halfDim;
  
    return [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
};

/**
 * Crop a tensor based on a given bounding box
 * @param {tf.Tensor} tensor The input tensor (image)
 * @param {Array} boundingBox The bounding box [y1, x1, y2, x2]
 * @returns {tf.Tensor} Cropped tensor
 */
export const cropTensor = (tensor, boundingBox) => {
    return tf.tidy(() => {
      const [x1, y1, x2, y2] = boundingBox;
  
      // Calculate the height and width of the bounding box
      const height = Math.abs(y2 - y1);
      const width = Math.abs(x2 - x1);
  
      // Slice the tensor to get the cropped region
      const cropped = tensor.slice([y1, x1, 0], [height, width, tensor.shape[2]]);
  
      return cropped;
    });
  };

/**
 * Convert a tensor to ImageData
 * @param {tf.Tensor} tensor The input tensor (image) with shape [height, width, channels]
 * @param {boolean} [isNormalized=false] Flag indicating if the tensor values are normalized (range 0-1)
 * @returns {Promise<ImageData>} A promise that resolves to an ImageData object
 */
export const convertTensorToImageData = async (tensor, isNormalized = false) => {
    // Ensure the tensor is 3-dimensional
    if (tensor.rank !== 3) {
      throw new Error('Tensor must be of shape [height, width, channels]');
    }
  
    const [height, width, numChannels] = tensor.shape;
    const tensorData = await tensor.data(); // Get the data from the tensor
  
    // Create an ImageData object
    const imageData = new ImageData(width, height);
    for (let i = 0; i < height * width; i++) {
      const j = i * 4; // index in imageData
      const k = i * numChannels; // index in tensorData
      const multiplier = isNormalized ? 255 : 1;
      imageData.data[j] = tensorData[k] * multiplier; // R (or B if tensor is in BGR)
      imageData.data[j + 1] = tensorData[k + 1] * multiplier; // G
      imageData.data[j + 2] = tensorData[k + 2] * multiplier; // B (or R if tensor is in BGR)
      imageData.data[j + 3] = 255; // A (fully opaque)
    }
  
    return imageData;
  };  
  
  /**
 * Convert a tensor to a downloadable image
 * @param {tf.Tensor} tensor The input tensor (image)
 * @param {string} filename The name of the downloaded file
 */
export const tensorToDownloadableImage = async (tensor, filename = 'image.png') => {
    // Ensure the tensor is in CPU memory
    const tensorData = await tensor.data();
  
    // Create a canvas element
    const [batch, height, width, numChannels] = tensor.shape;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
  
    // Create image data from the tensor
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < height * width; i++) {
      const j = i * 4;
      const k = i * numChannels;
      imageData.data[j] = tensorData[k] * 255;      // R
      imageData.data[j + 1] = tensorData[k + 1] * 255; // G
      imageData.data[j + 2] = tensorData[k + 2] * 255; // B
      imageData.data[j + 3] = 255;            // A
    }
  
    // Put the image data on the canvas
    ctx.putImageData(imageData, 0, 0);
  
    // Convert the canvas to a data URL
    const dataURL = canvas.toDataURL('image/png');
  
    // Create a download link
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = filename;
    document.body.appendChild(link);
  
    // Trigger the download
    link.click();
  
    // Remove the link from the document
    document.body.removeChild(link);
  };