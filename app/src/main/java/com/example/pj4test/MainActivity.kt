package com.example.pj4test

import android.Manifest.permission.CAMERA
import android.Manifest.permission.RECORD_AUDIO
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.example.pj4test.cameraInference.PlantHealthClassifier
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.*

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"

    // permissions
    private val permissions = arrayOf(RECORD_AUDIO, CAMERA)
    private val PERMISSIONS_REQUEST = 0x0000001;

    // Create an instance of the PlantHealthClassifier
    private val plantHealthClassifier = PlantHealthClassifier()

    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        checkPermissions() // check permissions

        // Initialize the PlantHealthClassifier
        plantHealthClassifier.initialize(this)

        // Set the detector listener
        plantHealthClassifier.setDetectorListener(object : PlantHealthClassifier.DetectorListener {
            override fun onObjectDetectionError(error: String) {
                // Handle the error
            }

            override fun onObjectDetectionResults(
                results: MutableList<Detection>?,
                inferenceTime: Long,
                imageHeight: Int,
                imageWidth: Int
            ) {
                // Handle the detection results
            }
        })
    }

    private fun checkPermissions() {
        if (permissions.all { ActivityCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            Log.d(TAG, "All Permission Granted")
        } else {
            requestPermissions(permissions, PERMISSIONS_REQUEST)
        }
    }
}

