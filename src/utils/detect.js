import * as tf from "@tensorflow/tfjs";
import { renderBoxes, createMaskedFrame } from "./renderBox";
import { tensorToDownloadableImage, cropTensor, scaleAndPositionBoundingBox } from "./tensor";
import labels from "./labels.json";

const numClass = labels.length;

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @param {Array} faceBox The bounding box [x1, y1, x2, y2]
 * @returns input tensor, xRatio and yRatio
 */
const preprocess = (source, modelWidth, modelHeight, faceBox) => {
  let xRatio, yRatio; // ratios for boxes

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    const croppedImg = faceBox ? cropTensor(img, faceBox) : img;
    const [originalH, originalW] = croppedImg.shape.slice(0, 2); // get source width and height

    // Get source dimensions
    const [h, w] = croppedImg.shape.slice(0, 2); // get source width and height
    const aspectRatio = w / h; // calculate aspect ratio

    // Calculate new dimensions while maintaining aspect ratio
    let newWidth, newHeight;
    if (w > h) {
      newWidth = modelWidth;
      newHeight = modelWidth / aspectRatio;
    } else {
      newHeight = modelHeight;
      newWidth = modelHeight * aspectRatio;
    }

    // Resize the image to the new dimensions
    const imgResized = tf.image.resizeBilinear(croppedImg, [Math.round(newHeight), Math.round(newWidth)]);
    xRatio = modelWidth / w; // update xRatio based on the original width
    yRatio = modelHeight / h; // update yRatio based on the original height

    // Calculate padding to fit the resized image into the model's input dimensions
    const padHeight = modelHeight - Math.round(newHeight);
    const padWidth = modelWidth - Math.round(newWidth);

    // Add padding to the resized image to fit the model dimensions
    const imgPadded = imgResized.pad([
      [Math.floor(padHeight / 2), Math.ceil(padHeight / 2)], // padding y [top, bottom]
      [Math.floor(padWidth / 2), Math.ceil(padWidth / 2)],   // padding x [left, right]
      [0, 0],
    ]);

    return imgPadded.div(255.0).expandDims(0); // normalize and add batch dimension
  });

  return [input, xRatio, yRatio];
};

/**
 * Function run inference and do detection from source.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {VoidFunction} callback function to run after detection process
 * @param {Boolean} useMask whether to use createMaskedFrame or renderBoxes
 */
export const detect = async (source, model, canvasRef, callback = () => { }, useMask = false, faceBox = null) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  // Get the original image dimensions
  const originalWidth = source.width || source.videoWidth;
  const originalHeight = source.height || source.videoHeight;

  tf.engine().startScope(); // start scoping tf engine

  const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight, faceBox); // preprocess image
  
  const res = model.net.execute(input); // inference model
  console.log("model result", res);

  const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
    const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
    const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
    const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), //y2
          tf.add(x1, w), //x2
        ],
        2
      )
      .squeeze();
  }); // process boxes [y1, x1, y2, x2]

  const [scores, classes] = tf.tidy(() => {
    // class scores
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0);
    return [rawScores.max(1), rawScores.argMax(1)];
  }); // get max scores and classes index

  const boxes_data = boxes.dataSync(); // get boxes data
  const scores_data = scores.dataSync(); // get scores data
  const classes_data = classes.dataSync(); // get classes

  // Filter arrays
  let filteredIndices = [];
  let maxConfIndex = -1;
  let maxConf = -1;

  for (let i = 0; i < classes_data.length; i++) {
    if (classes_data[i] === 0) {
      if (scores_data[i] > maxConf) {
        maxConf = scores_data[i];
        maxConfIndex = i;
      }
      filteredIndices.push(i);
    }
  }

  // If we found any class 1 detections, keep only the one with highest confidence
  if (maxConfIndex !== -1) {
    filteredIndices = [maxConfIndex];
  }

  if (filteredIndices.length === 0) {
    // tensorToDownloadableImage(input, "input.png");
  }

  // Create new filtered arrays
  const filtered_boxes_data = filteredIndices.reduce((acc, i) => {
    const box = boxes_data.slice(i * 4, (i + 1) * 4);
    const adjusted_box = scaleAndPositionBoundingBox(box, xRatio, yRatio, faceBox, originalWidth, originalHeight);
    acc.push(...adjusted_box);
    return acc;
  }, []);
  const filtered_scores_data = filteredIndices.map(i => scores_data[i]);
  const filtered_classes_data = filteredIndices.map(i => classes_data[i]);

  // Replace the renderBoxes call with this conditional block
  if (useMask) {
    createMaskedFrame(canvasRef, filtered_boxes_data, filtered_scores_data, filtered_classes_data, [1, 1], source);
  } else {
    renderBoxes(canvasRef, filtered_boxes_data, filtered_scores_data, filtered_classes_data, [1, 1], source);
  }

  tf.dispose([res, transRes, boxes, scores, classes]); // clear memory

  callback();

  tf.engine().endScope(); // end of scoping
};

/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {Boolean} useMask whether to use createMaskedFrame or renderBoxes
 */
export const detectVideo = (vidSource, model, canvasRef, useMask = false) => {
  /**
   * Function to detect every frame from video
   */
  const detectFrame = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      const ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
      return; // handle if source is closed
    }

    detect(vidSource, model, canvasRef, () => {
      requestAnimationFrame(detectFrame); // get another frame
    }, useMask);
  };

  detectFrame(); // initialize to detect every frame
};
