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
    private lateinit var lightspeech: OrtSession
    private lateinit var mbmelgan: OrtSession
    private var resultText : TextView? = null
    private var inferenceButton: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inferenceButton = findViewById(R.id.inference_button)
        resultText = findViewById(R.id.result_text)

        // initialize sessions / load models
        val lightspeechPath = resources.openRawResource(R.raw.lightspeech_quant).readBytes()
        val mbmelganPath = resources.openRawResource(R.raw.mbmelgan).readBytes()
        lightspeech = ortEnv.createSession(lightspeechPath, sessionOptions)
        mbmelgan = ortEnv.createSession(mbmelganPath, sessionOptions)

        inferenceButton?.setOnClickListener {
            try {
                performInference(lightspeech, mbmelgan)
                Toast.makeText(baseContext, "Inference Successful!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when performing inference", e)
                Toast.makeText(baseContext, "Failed to perform inference", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun performInference(lightspeechSession: OrtSession, mbmelganSession: OrtSession) {
        val start = System.nanoTime()

        val lightspeech = LightSpeech()
        val mbmelgan = MBMelGAN()

        // infer; LightSpeech returns 3 outputs: (mel, duration, pitch)
        val lightspeechResults = lightspeech.infer(ortEnv, lightspeechSession)
        // NOTE: FastSpeech2 returns >3 outputs!

        // NOTE: this is durations for visemes!
        val durations = lightspeechResults.durations[0]

        // infer melgan
        val mels = lightspeechResults.mels
        val mbmelganResults = mbmelgan.infer(ortEnv, mbmelganSession, mels)

        var durationString = ""
        for (i in durations) {
            durationString += i
            durationString += " "
        }

        val inferenceTime = ((System.nanoTime() - start) / 1_000_000).toString()
        val outputText = "Inference time: $inferenceTime ms\nDurations: $durationString"
        resultText?.setText(outputText)

        val audio = mbmelganResults.audio[0].flatMap { it.asIterable() }.toFloatArray()

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
        lightspeech.close()
        mbmelgan.close()
    }

    companion object {
        private const val TAG = "ORTSuperResolution"
        private const val SAMPLE_RATE = 44100
        private const val FORMAT = AudioFormat.ENCODING_PCM_FLOAT
        private const val CHANNEL = AudioFormat.CHANNEL_OUT_MONO
    }

}