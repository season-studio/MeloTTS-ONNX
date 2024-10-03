# MeloTTS by ONNX
[中文](./README_cn.md)

## Introduction
This is an implementation of Melo TTS by onnxruntime.
We restruct the text utils for speeding up.
We convert the models into onnx format. The models are stored in this repository as lfs.

We has just implement the zh-mix-en language. Other languages will come soon.

## Usage

```python

from melo_onnx import MeloTTX_ONNX
import soundfile

model_path = "path/to/folder/of/model"
tts = MeloTTX_ONNX(model_path)
audio = tts.speak("今天天气真nice。", tts.speakers[0])

soundfile.write("path/of/audio.wav", audio, samplerate=tts.sample_rate)

```

The parameters of the constructor of MeloTTX_ONNX:
- **model_path** str. The path of the folder store the model.
- **execution_provider** str. The device for the onnxruntime, CUDA, CPU, or others. If it's None, the library will choose the better one between the CUDA and CPU.
- **verbose** bool. Set True for display the detail information when the library working.
- **onnx_session_options** onnxruntim.SessionOptions. You can setup the special options for the onnx session.
- **onnx_params** dict. The other parameters you want to pass into the onnxruntim.InferenceSession 

The parameters of the MeloTTX_ONNX.speak:
- **text** str. The text you want to synthesis.
- **speaker** str. The speaker you want to use.
- **speed** float. The speed of the speech
- **sdp_ratio** float.
- **noise_scale** float.
- **noise_scale_w** float.
- **pbar** function. Such as tqdm

Some useful property of the instance of the MeloTTS:
- **speakers**: [str]. Readonly. The available speakers
- **sample_rate**: int. Readonly. The sample rate of the synthesis result
- **language**: str. Readonly. The language of the current model