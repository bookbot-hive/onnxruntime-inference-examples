// Modified from: https://github.com/ZDisket/TensorVox

// ONNX Runtime header
#include <onnxruntime_cxx_api.h>

// C++ headers
#include <iostream>
#include <sstream>

// AudioFile headers
// #include "AudioFile.hpp"

// void ExportWAV(const std::string &Filename, const std::vector<float> &Data, unsigned SampleRate)
// {
//     AudioFile<float>::AudioBuffer Buffer;
//     Buffer.resize(1);

//     Buffer[0] = Data;
//     size_t BufSz = Data.size();

//     AudioFile<float> File;

//     File.setAudioBuffer(Buffer);
//     File.setAudioBufferSize(1, (int)BufSz);
//     File.setNumSamplesPerChannel((int)BufSz);
//     File.setNumChannels(1);
//     File.setBitDepth(32);
//     File.setSampleRate(SampleRate);

//     File.save(Filename, AudioFileFormat::Wave);
// }

std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

using namespace std;

int main()
{   
    const char* model_path = "/root/lightspeech-mfa-en-v6/lightspeech_quant.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
    Ort::SessionOptions session_options;
    Ort::Session session = Ort::Session(env, model_path, session_options);  // access experimental components via the Experimental namespace

    // raw input vectors
    std::vector<int32_t> input_ids = {25, 29, 13, 40, 17, 51, 23, 29, 17, 12, 42, 16, 51, 14, 8, 51, 23, 3, 50, 4, 71, 68, 14, 29, 22, 50, 34, 29, 21, 25, 29, 4, 42, 21, 9, 29, 17, 17, 16, 51, 34, 33, 18, 17, 18, 47, 11, 33, 26, 8, 51, 13, 51, 14, 25, 29, 14, 39, 18, 72};
    std::vector<float> energy_ratios = {1.f};
    std::vector<float> f0_ratios = {1.f};
    std::vector<int32_t> speaker_ids = {2};
    std::vector<float> speed_ratios = {1.f};

    // This is the shape of the inputs, our equivalent to tf.expand_dims.
    std::vector<int64_t> input_ids_shape = {1, (int64_t)input_ids.size()};
    std::vector<int64_t> energy_ratios_shape = {1};
    std::vector<int64_t> f0_ratios_shape = {1};
    std::vector<int64_t> speed_ratios_shape = {1};
    std::vector<int64_t> speaker_ids_shape = {1};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);   

    // std::vector<Ort::Value> input_tensors;
    // input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size()));
    // input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, speaker_ids.data(), speaker_ids.size(), speaker_ids_shape.data(), speaker_ids_shape.size()));
    // input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, speed_ratios.data(), speed_ratios.size(), speed_ratios_shape.data(), speed_ratios_shape.size()));
    // input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, f0_ratios.data(), f0_ratios.size(), f0_ratios_shape.data(), f0_ratios_shape.size()));
    // input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, energy_ratios.data(), energy_ratios.size(), energy_ratios_shape.data(), energy_ratios_shape.size()));

    // Define the tensors
    // Ort::Value {variable_name} = Ort::Value::CreateTensor<dtype>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
    // TODO: change speaker index here
    Ort::Value speaker_ids_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, speaker_ids.data(), speaker_ids.size(), speaker_ids_shape.data(), speaker_ids_shape.size());
    Ort::Value speed_ratios_tensor = Ort::Value::CreateTensor<float>(memory_info, speed_ratios.data(), speed_ratios.size(), speed_ratios_shape.data(), speed_ratios_shape.size());
    Ort::Value f0_ratios_tensor = Ort::Value::CreateTensor<float>(memory_info, f0_ratios.data(), f0_ratios.size(), f0_ratios_shape.data(), f0_ratios_shape.size());
    Ort::Value energy_ratios_tensor = Ort::Value::CreateTensor<float>(memory_info, energy_ratios.data(), energy_ratios.size(), energy_ratios_shape.data(), energy_ratios_shape.size());

    const char* input_names[] = {"input_ids", "speaker_ids", "speed_ratios", "f0_ratios", "energy_ratios"};
    const char* output_names[] = {"Identity", "Identity_1", "Identity_2"};

    std::vector<std::reference_wrapper<Ort::Value>> input_tensors = { input_ids_tensor, speaker_ids_tensor, speed_ratios_tensor, f0_ratios_tensor, energy_ratios_tensor };

    std::vector<const Ort::Value*> inputPointers;
    for (const auto& inputValue : input_tensors) {
        inputPointers.push_back(&inputValue.get());
    }

    // run inference
    Ort::RunOptions run_options;
    // infer; LightSpeech returns 3 outputs: (mel, duration, pitch)
    std::vector<Ort::Value> outputs = session.Run(run_options, input_names, inputPointers.data(), (size_t)5, output_names, (size_t)3);
    // NOTE: FastSpeech2 returns >3 outputs!

    // TFTensor<float> mel_spec = CopyTensor<float>(outputs[0]);
    // TFTensor<int32_t> durations = CopyTensor<int32_t>(outputs[1]);

    // // prepare mel spectrograms for input
    // cppflow::tensor input_mels{mel_spec.Data, mel_spec.Shape};
    // // infer
    // auto out_audio = mbmelgan({{"serving_default_mels:0", input_mels}}, {"StatefulPartitionedCall:0"})[0];
    // TFTensor<float> audio_tensor = CopyTensor<float>(out_audio);

    // // write to file, specify sample rate
    // ExportWAV("output.wav", audio_tensor.Data, 44100);

    return 0;
}