package com.example.speechsynthesis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import java.util.*


internal data class Result(
    var durations: String = "",
    var audio: FloatArray = floatArrayOf(),
    var inferenceTime: String = ""
) {}


internal class JETS(
) {

    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        val result = Result()

        val ids = longArrayOf(6, 31, 6, 19, 14, 43, 13, 36, 26, 27, 7)

        val start = System.nanoTime()
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            LongBuffer.wrap(ids),
            longArrayOf(ids.size.toLong()),
        )
        inputTensor.use {
            val output = ortSession.run(Collections.singletonMap("text", inputTensor))

            output.use {
                result.inferenceTime = ((System.nanoTime() - start) / 1_000_000).toString()
                val audio = (output?.get(0)?.value) as FloatArray
                val durations = (output?.get(1)?.value) as LongArray

                var durationString = "";
                for (i in durations) {
                    durationString += i
                    durationString += " "
                }
                result.durations = durationString
                result.audio = audio

            }
        }
        return result
    }
}
