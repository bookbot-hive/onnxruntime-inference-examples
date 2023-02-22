package com.example.speechsynthesis

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.min


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var resultText : TextView? = null
    private var inferenceButton: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inferenceButton = findViewById(R.id.inference_button)
        resultText = findViewById(R.id.result_text)

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        inferenceButton?.setOnClickListener {
            try {
                performInference(ortSession)
                Toast.makeText(baseContext, "Inference Successful!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when performing inference", e)
                Toast.makeText(baseContext, "Failed to perform inference", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun performInference(ortSession: OrtSession) {
        var jets = JETS()
        var results = jets.infer(ortEnv, ortSession)

        var durations = results.durations
        var inferenceTime = results.inferenceTime
        val outputText = "Inference time: $inferenceTime ms\nDurations: $durations"
        resultText?.setText(outputText)

        var audio = results.audio

        val bufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL, FORMAT)
        val audioTrack: AudioTrack = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                .build(),
            AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setEncoding(FORMAT)
                .setChannelMask(CHANNEL)
                .build(),
            bufferSize,
            AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE
        )
        var index = 0
        audioTrack.play()
        while (index < audio.size) {
            val buffer = min(bufferSize, audio.size - index)
            audioTrack.write(
                audio,
                index,
                buffer,
                AudioTrack.WRITE_BLOCKING
            )
            index += bufferSize
        }
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

    companion object {
        private const val TAG = "ORTSuperResolution"
        private const val SAMPLE_RATE = 22050
        private const val FORMAT = AudioFormat.ENCODING_PCM_FLOAT
        private const val CHANNEL = AudioFormat.CHANNEL_OUT_MONO

    }

}