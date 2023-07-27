package com.example.speechsynthesis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer


internal data class PiperResult(
    var audio: Array<Array<Array<FloatArray>>>
) {}


internal class Piper(
) {
    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession): PiperResult {
        val result: PiperResult

        val input = longArrayOf(0, 64, 0, 156, 0, 102, 0, 62, 0, 61, 0, 16, 0, 102, 0, 68, 0, 16, 0, 156, 0, 76, 0, 158, 0, 61, 0, 138, 0, 55, 0, 5, 0)
        val inputLength = longArrayOf(input.size.toLong())
        val scales = floatArrayOf(0.667F, 1.0F, 0.8F)

        // this is the shape of the inputs, our equivalent to tf.expand_dims.
        val inputShape = longArrayOf(1, input.size.toLong())
        val inputLengthShape = longArrayOf(1)
        val scalesShape = longArrayOf(3)

        val inputNames = arrayOf("input", "input_lengths", "scales")

        // create input tensors from raw vectors
        val inputTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(input), inputShape)
        val inputLengthTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputLength), inputLengthShape)
        val scalesTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(scales), scalesShape)

        val inputTensorsVector = arrayOf(inputTensor, inputLengthTensor, scalesTensor)

        // create input name -> input tensor map
        val inputTensors: Map<String, OnnxTensor> = inputNames.zip(inputTensorsVector).toMap()

        val output = ortSession.run(inputTensors)
        output.use {
            val audio = (output?.get(0)?.value) as Array<Array<Array<FloatArray>>> // (1, frames) => (1, acousticFrames * hopLength)

            result = PiperResult(audio)
        }
        return result
    }
}
