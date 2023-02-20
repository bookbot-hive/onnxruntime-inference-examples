package com.example.speechsynthesis

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.*


internal data class Result(
    var outputResult:String = ""
) {}


internal class JETS(
) {

    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        val result = Result()


        val ids = longArrayOf(6, 31, 6, 19, 14, 43, 13, 36, 26, 27, 7)
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            LongBuffer.wrap(ids),
            longArrayOf(ids.size.toLong()),
        )
        inputTensor.use {
            val output = ortSession.run(Collections.singletonMap("text", inputTensor))

            output.use {
                val rawOutput = (output?.get(1)?.value) as LongArray
                var output = "";
                for (i in rawOutput) {
                    output += i
                    output += " "
                }
                result.outputResult = output
            }
        }
        return result
    }
}
