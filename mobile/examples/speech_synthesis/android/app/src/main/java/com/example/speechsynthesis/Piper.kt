package com.example.speechsynthesis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer


internal data class PiperResult(
    var audio: Array<Array<FloatArray>>,
    var durations: FloatArray
) {}


internal class Piper(
) {
    fun sumVertically(array: Array<FloatArray>): FloatArray {
        val numRows = array.size
        val numCols = array[0].size

        val sumArray = FloatArray(numCols)

        for (col in 0 until numCols) {
            var sum = 0.0f
            for (row in 0 until numRows) {
                sum += array[row][col]
            }
            sumArray[col] = sum
        }

        return sumArray
    }

    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession): PiperResult {
        val result: PiperResult

        val input = longArrayOf(1,  0, 17,  0, 10,  0, 21,  0, 24,  0,  9,  0, 13,  0, 29,  0, 23,  0, 18,  0, 10,  0,  9,  0,  6,  0,  2)
        val inputLength = longArrayOf(input.size.toLong())
        val scales = floatArrayOf(0.667F, 1.0F, 0.8F)
        // TODO: change if multispeaker! only `null` if single-speaker
        val sid = null

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
            val audio = (output?.get(0)?.value) as Array<Array<FloatArray>> // (1, 1, frames) => (1, 1, acousticFrames * hopLength)
            val attention = (output?.get(1)?.value) as Array<Array<Array<FloatArray>>> // (1, 1, acousticFrames, inputLength)

            val durations = sumVertically(attention[0][0])
            result = PiperResult(audio, durations)
        }
        return result
    }
}
