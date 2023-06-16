package com.example.pj4test.cameraInference

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class PlantHealthClassifier {
    // Libraries for object detection
    lateinit var objectDetector: ObjectDetector

    // Listener that will be handle the result of this classifier
    private var objectDetectorListener: DetectorListener? = null

    fun initialize(context: Context) {
        setupObjectDetector(context)
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    private fun setupObjectDetector(context: Context) {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(THRESHOLD)
                .setMaxResults(MAX_RESULTS)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(NUM_THREADS)
        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, MODEL_NAME, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onObjectDetectionError(
                "Object detector failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        objectDetectorListener?.onObjectDetectionResults(
            results,
            inferenceTime,
            tensorImage.height,
            tensorImage.width)
    }

    interface DetectorListener {
        fun onObjectDetectionError(error: String)
        fun onObjectDetectionResults(
            results: MutableList<Detection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    fun setDetectorListener(listener: DetectorListener) {
        objectDetectorListener = listener
    }

    companion object {
        const val THRESHOLD: Float = 0.5f
        const val NUM_THREADS: Int = 2
        const val MAX_RESULTS: Int = 3
        const val MODEL_NAME = "plant_health_model.tflite"
    }
}
