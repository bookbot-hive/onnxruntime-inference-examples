package com.example.speechsynthesis

import ai.onnxruntime.*
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
    private var sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
    private lateinit var vits: OrtSession
    private var resultText : TextView? = null
    private var inferenceButton: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inferenceButton = findViewById(R.id.inference_button)
        resultText = findViewById(R.id.result_text)

        // initialize sessions / load models
        val vitsPath = resources.openRawResource(R.raw.vits).readBytes()
        vits = ortEnv.createSession(vitsPath, sessionOptions)

        inferenceButton?.setOnClickListener {
            try {
                performInference(vits)
                Toast.makeText(baseContext, "Inference Successful!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when performing inference", e)
                Toast.makeText(baseContext, "Failed to perform inference", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun performInference(vitsSession: OrtSession) {
        val start = System.nanoTime()

        val vits = VITS()

        // infer; LightSpeech returns 3 outputs: (mel, duration, pitch)
        val vitsResults = vits.infer(ortEnv, vitsSession)
        // NOTE: FastSpeech2 returns >3 outputs!

        // NOTE: this is durations for visemes!
        // val durations = lightspeechResults.durations[0]

         var durationString = ""
        // for (i in durations) {
        //     durationString += i
        //     durationString += " "
        // }

        val inferenceTime = ((System.nanoTime() - start) / 1_000_000).toString()
        val outputText = "Inference time: $inferenceTime ms\nDurations: $durationString"
        resultText?.setText(outputText)

        val audio = vitsResults.audio[0][0]

        val bufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL, FORMAT)
        val audioTrack = AudioTrack(
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

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        vits.close()
    }

    companion object {
        private const val TAG = "ORTSuperResolution"
        private const val SAMPLE_RATE = 22050
        private const val FORMAT = AudioFormat.ENCODING_PCM_FLOAT
        private const val CHANNEL = AudioFormat.CHANNEL_OUT_MONO
    }

}