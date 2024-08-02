/**
 * Class to handle webcam
 */

import { WIDTH, HEIGHT, FRAME_RATE } from "../consts";

const constraints = {
  audio: false,
  video: {
    width: { exact: WIDTH },
    height: { exact: HEIGHT },
    frameRate: { exact: FRAME_RATE },
    facingMode: "user",
  },
};

export class Webcam {
  /**
   * Open webcam and stream it through video tag.
   * @param {HTMLVideoElement} videoRef video tag reference
   */
  open = (videoRef) => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia(constraints)
        .then((stream) => {
          videoRef.srcObject = stream;
          videoRef.width = WIDTH;
          videoRef.height = HEIGHT;
        });
    } else alert("Can't open Webcam!");
  };

  /**
   * Close opened webcam.
   * @param {HTMLVideoElement} videoRef video tag reference
   */
  close = (videoRef) => {
    if (videoRef.srcObject) {
      videoRef.srcObject.getTracks().forEach((track) => {
        track.stop();
      });
      videoRef.srcObject = null;
    } else alert("Please open Webcam first!");
  };
}
