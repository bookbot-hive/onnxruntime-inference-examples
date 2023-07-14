package com.example.speechsynthesis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.*


internal data class MBMelGANOutputs(
    val audio: Array<Array<FloatArray>>
)

internal class MBMelGAN(
) {

    fun infer(ortEnv: OrtEnvironment, ortSession: OrtSession, mels: Array<Array<FloatArray>>): MBMelGANOutputs {
        val result: MBMelGANOutputs

        // unpack 3d FloatArray and get size along each dimension := (1, L, 80)
        val melsShape = longArrayOf(mels.size.toLong(), mels[0].size.toLong(), mels[0][0].size.toLong())

        val totalElements = mels.size * mels[0].size * mels[0][0].size
        val flattenedMels = FloatArray(totalElements) { index ->
            val i = index / (mels[0].size * mels[0][0].size)
            val j = (index % (mels[0].size * mels[0][0].size)) / mels[0][0].size
            val k = (index % (mels[0].size * mels[0][0].size)) % mels[0][0].size
            mels[i][j][k]
        }

        // create input tensors from raw vectors
        val melTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(flattenedMels), melsShape)

        // create input name -> input tensor map
        val inputTensors: Map<String, OnnxTensor> = Collections.singletonMap("mels", melTensor)

        val output = ortSession.run(inputTensors)
        output.use {
            val audio = output?.get(0)?.value as Array<Array<FloatArray>>
            result = MBMelGANOutputs(audio)
        }

        return result
    }
}
