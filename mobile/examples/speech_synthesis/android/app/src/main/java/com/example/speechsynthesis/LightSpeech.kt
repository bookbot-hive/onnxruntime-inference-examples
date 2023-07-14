package com.example.speechsynthesis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.IntBuffer


internal data class LightSpeechOutputs(
    val mels: Array<Array<FloatArray>>,
    val durations: Array<IntArray>
)


internal class LightSpeech(
) {

    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession): LightSpeechOutputs {
        val result: LightSpeechOutputs

        // raw input vectors
        val inputIDs = intArrayOf(25, 29, 13, 40, 17, 51, 23, 29, 17, 12, 42, 16, 51, 14, 8, 51, 23, 3, 50, 4, 71, 68, 14, 29, 22, 50, 34, 29, 21, 25, 29, 4, 42, 21, 9, 29, 17, 17, 16, 51, 34, 33, 18, 17, 18, 47, 11, 33, 26, 8, 51, 13, 51, 14, 25, 29, 14, 39, 18, 72)
        // TODO: change speaker index here
        val speakerIDs = intArrayOf(0)
        val speedRatios = floatArrayOf(1.0F)
        val f0Ratios = floatArrayOf(1.0F)
        val energyRatios = floatArrayOf(1.0F)

        // this is the shape of the inputs, our equivalent to tf.expand_dims.
        val inputIDsShape = longArrayOf(1, inputIDs.size.toLong())
        val speakerIDsShape = longArrayOf(1)
        val speedRatiosShape = longArrayOf(1)
        val f0RatiosShape = longArrayOf(1)
        val energyRatiosShape = longArrayOf(1)

        val inputNames = arrayOf("input_ids", "speaker_ids", "speed_ratios", "f0_ratios", "energy_ratios")

        // create input tensors from raw vectors
        val inputIDsTensor = OnnxTensor.createTensor(ortEnv, IntBuffer.wrap(inputIDs), inputIDsShape)
        val speakerIDsTensor = OnnxTensor.createTensor(ortEnv, IntBuffer.wrap(speakerIDs), speakerIDsShape)
        val speedRatiosTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(speedRatios), speedRatiosShape)
        val f0RatiosTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(f0Ratios), f0RatiosShape)
        val energyRatiosTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(energyRatios), energyRatiosShape)
        val inputTensorsVector = arrayOf(inputIDsTensor, speakerIDsTensor, speedRatiosTensor, f0RatiosTensor, energyRatiosTensor)

        // create input name -> input tensor map
        val inputTensors: Map<String, OnnxTensor> = inputNames.zip(inputTensorsVector).toMap()

        val output = ortSession.run(inputTensors)
        output.use {
            val mels = output?.get(0)?.value as Array<Array<FloatArray>>
            val durations = output?.get(1)?.value as Array<IntArray>
            result = LightSpeechOutputs(mels, durations)
        }

        return result
    }
}
