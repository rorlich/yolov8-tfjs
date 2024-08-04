import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import "./style/App.css";
import * as Mp4Muxer from "mp4-muxer";
import { WIDTH, HEIGHT, FRAME_RATE } from "./consts";

let startTime = null;
let lastKeyFrame = null;
let framesGenerated = 0;
let recording = false;

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const muxerRef = useRef({ current: null });
  const videoEncoderRef = useRef({ current: null });
  const skippedFramesRef = useRef(0); // Ref to track skipped frames
  const skippedFramesDisplayRef = useRef(null); // Ref to the HTML element for displaying skipped frames
  const [framesSkippedCount, setFramesSkippedCount] = useState(0);
  // model configs
  const modelName = "yolov8n";

  let processIntervalId = null;
  
  const processStream = (vidSource, model, canvasRef) => {  
    initMuxer(); 
    
    let isProcessing = false; // Flag to track if processFrame is currently running

    // canvasRef.width = WIDTH
    // canvasRef.height = HEIGHT

    /**
     * Function to detect every frame from video
     */
    const processFrame = async () => {
      if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
        const ctx = canvasRef.getContext("2d");
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

        // stop the interval if source is closed
        clearInterval(processIntervalId);
        closeVideoEncoder(true);
        return; // handle if source is closed
      }

      if (canvasRef === null) {
        return; // handle if canvas is not ready
      }

      const timestamp = performance.now() * 1000;

      if (isProcessing) {
        setFramesSkippedCount((prev) => prev + 1);
        return; // Skip this interval if the previous frame is still processing
      }

      isProcessing = true;
      console.log("Processing at", new Date().toISOString());
    
      // Perform detection and any other processing here
      await detect(vidSource, model, canvasRef, () => {}, true);
    
      // Encode the current content of the canvas as a video frame
      await encodeVideoFrame(canvasRef, timestamp);
    
      isProcessing = false;
    
      // processTimeoutId = setTimeout(processFrame, Math.ceil(1000 / 15));
    };
  
    processIntervalId = setInterval(processFrame, Math.ceil(1000 / 15));
  };

  const initMuxer = async () => {
    let muxer = new Mp4Muxer.Muxer({
      target: new Mp4Muxer.ArrayBufferTarget(),

      video: {
        codec: "avc",
        width:  WIDTH,
        height:  HEIGHT,
      },
      // Puts metadata to the start of the file. Since we're using ArrayBufferTarget anyway, this makes no difference
      // to memory footprint.
      fastStart: "in-memory",

      // Because we're directly pumping a MediaStreamTrack's data into it, which doesn't start at timestamp = 0
      firstTimestampBehavior: "offset",
    });

    let videoEncoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => console.error(e),
    });
    videoEncoder.configure({
      codec: "avc1.64001F",
      width:  WIDTH,
      height: HEIGHT,
      bitrate: 2_000_000, // 2 Mbps
      framerate: FRAME_RATE,
    });

    startTime = document.timeline.currentTime;
    recording = true;
    lastKeyFrame = -Infinity;
    framesGenerated = 0;

    muxerRef.current = muxer;
    videoEncoderRef.current = videoEncoder;
  };

  const encodeVideoFrame = async (refCurrent, timestamp) => {
    const elapsedTime = document?.timeline?.currentTime
      ? document.timeline.currentTime - startTime
      : startTime;

    const frame = new VideoFrame(refCurrent, {
      // timestamp: (framesGenerated * 1e6) / 30, // Ensure equally-spaced frames every 1/30th of a second
      // duration: 1e6 / 30,
      timestamp: timestamp,
    });
    framesGenerated++;

    // Ensure a video key frame at least every 5 seconds for good scrubbing
    let needsKeyFrame = elapsedTime - lastKeyFrame >= 5000;
    if (needsKeyFrame) {
      lastKeyFrame = elapsedTime;
    }
    if (videoEncoderRef.current) {
      videoEncoderRef.current.encode(frame, { keyFrame: needsKeyFrame });
    }
    frame.close();
  };

  const closeVideoEncoder = async (download) => {
    recording = false;
    if (videoEncoderRef.current) {
      await videoEncoderRef.current.flush();
    }
    await muxerRef.current?.finalize();
    let buffer = muxerRef.current?.target.buffer;

    if (download) {
      downloadBlob(new Blob([buffer]));
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const downloadBlob = (blob) => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = "HumanFaceDetection.mp4";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  };

  useEffect(() => {
    tf.ready().then(async () => {
      const yolov8 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model
      console.log(yolov8.inputs[0].shape);
      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      const warmupResults = yolov8.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov8,
        inputShape: yolov8.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    });
  }, []);

  return (
    <div className="App">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
      <div className="header">
        <h1>📷 YOLOv8 Live Detection App</h1>
        <p>
          YOLOv8 live detection application on browser powered by <code>tensorflow.js</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
        <p>
          Frames Skipped : <code className="code">{framesSkippedCount}</code>
        </p>

      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => detect(imageRef.current, model, canvasRef.current)}
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() => processStream(cameraRef.current, model, canvasRef.current)}
          onEndedCapture={() => console.log("Stopped")}
        />
        {/* <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() => processStream(videoRef.current, model, canvasRef.current)}
        /> */}
        <canvas width={model.inputShape[1]} height={model.inputShape[2]} ref={canvasRef} />
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
    </div>
  );
};

export default App;
