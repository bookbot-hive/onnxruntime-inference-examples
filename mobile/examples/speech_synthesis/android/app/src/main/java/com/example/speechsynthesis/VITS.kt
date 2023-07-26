package com.example.speechsynthesis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer


internal data class VITSResult(
    var audio: Array<Array<FloatArray>>
) {}


internal class VITS(
) {

    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession): VITSResult {
        val result: VITSResult

        val input = longArrayOf(0, 64, 0, 156, 0, 102, 0, 62, 0, 61, 0, 16, 0, 102, 0, 68, 0, 16, 0, 156, 0, 76, 0, 158, 0, 61, 0, 138, 0, 55, 0, 5, 0)
        val inputLength = longArrayOf(input.size.toLong())
        val noiseScale = floatArrayOf(0.667F)
        val lengthScale = floatArrayOf(1.0F)
        val noiseScaleW = floatArrayOf(0.8F)
        // TODO: change speaker index here
        // val speakerID = null

        // this is the shape of the inputs, our equivalent to tf.expand_dims.
        val inputShape = longArrayOf(1, input.size.toLong())
        val inputLengthShape = longArrayOf(1)
        val noiseScaleShape = longArrayOf(1)
        val lengthScaleShape = longArrayOf(1)
        val noiseScaleWShape = longArrayOf(1)
        // val speakerIDShape = longArrayOf(1)

        // val inputNames = arrayOf("input", "input_lengths", "noise_scale", "length_scale", "noise_scale_w", "speaker_id")
        val inputNames = arrayOf("input", "input_lengths", "noise_scale", "length_scale", "noise_scale_w")

        // create input tensors from raw vectors
        val inputTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(input), inputShape)
        val inputLengthTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputLength), inputLengthShape)
        val noiseScaleTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(noiseScale), noiseScaleShape)
        val lengthScaleTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(lengthScale), lengthScaleShape)
        val noiseScaleWTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(noiseScaleW), noiseScaleWShape)
        // val speakerIDTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(speakerID), speakerIDShape)

        // val inputTensorsVector = arrayOf(inputTensor, inputLengthTensor, noiseScaleTensor, lengthScaleTensor, noiseScaleWTensor, speakerIDTensor)
        val inputTensorsVector = arrayOf(inputTensor, inputLengthTensor, noiseScaleTensor, lengthScaleTensor, noiseScaleWTensor)

        // create input name -> input tensor map
        val inputTensors: Map<String, OnnxTensor> = inputNames.zip(inputTensorsVector).toMap()

        val output = ortSession.run(inputTensors)
        output.use {
            val audio = (output?.get(0)?.value) as Array<Array<FloatArray>>

            result = VITSResult(audio)
        }
        return result
    }
}
