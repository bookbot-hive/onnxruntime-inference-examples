package com.example.speechsynthesis

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var resultText : TextView? = null
    private var inferenceButton: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

//        inputImage = findViewById(R.id.imageView1)
//        outputImage = findViewById(R.id.imageView2);
        inferenceButton = findViewById(R.id.inference_button)
        resultText = findViewById(R.id.result_text)
//        inputImage?.setImageBitmap(
//            BitmapFactory.decodeStream(readInputImage())
//        );

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        inferenceButton?.setOnClickListener {
            try {
                performInference(ortSession)
                Toast.makeText(baseContext, "Super resolution performed!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform super resolution", e)
                Toast.makeText(baseContext, "Failed to perform super resolution", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    companion object {
        const val TAG = "ORTSuperResolution"
    }

    private fun performInference(ortSession: OrtSession) {
        var jets = JETS()
        var result = jets.infer(ortEnv, ortSession).outputResult
        resultText?.setText(result)
//        var result = superResPerformer.upscale(readInputImage(), ortEnv, ortSession)
//        updateUI(result);
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.model
        return resources.openRawResource(modelID).readBytes()
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

}