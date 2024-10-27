# MeloTTS and OpenVoice Tone Clone by ONNX
[中文](./README_cn.md)

## Introduction
This is an implementation of Melo TTS and OpenVoice Tone Clone by onnxruntime.
We restruct the text utils for speeding up.
We convert the models into onnx format. The models are stored in this repository as lfs.

We has just implement the zh-mix-en language. Other languages will come soon.

## Usage

```python
# tts demo
from melo_onnx import MeloTTX_ONNX
import soundfile

model_path = "path/to/folder/of/model_tts"
tts = MeloTTX_ONNX(model_path)
audio = tts.speak("今天天气真nice。", tts.speakers[0])

soundfile.write("path/of/result.wav", audio, samplerate=tts.sample_rate)

```

```python
# Tone clone demo
from melo_onnx import OpenVoiceToneClone_ONNX
tc = OpenVoiceToneClone_ONNX("path/to/folder/of/model_tone_clone")
import soundfile
tgt = soundfile.read("path/of/audio_for_tone_color", dtype='float32')
tgt = tc.resample(tgt[0], tgt[1])
tgt_tone_color = tc.extract_tone_color(tgt)
src = soundfile.read("path/of/audio_to_change_tone", dtype='float32')
src = tc.resample(src[0], src[1])
result = tc.tone_clone(src, tgt_tone_color)
soundfile.write("path/of/result.wav", result, tc.sample_rate)
```

**Models can be downloaded from the modelscope or huggingface:**
| model | repositories | comment |
|-------|--------------|---------|
| tts | [modelscope](https://modelscope.cn/models/seasonstudio/melotts_zh_mix_en_onnx) or [huggingface](https://huggingface.co/seasonstudio/melotts_zh_mix_en_onnx) | zh-mix-en |
| tone clone | [modelscope](https://modelscope.cn/models/seasonstudio/openvoice_tone_clone_onnx) or [huggingface](https://huggingface.co/seasonstudio/openvoice_tone_clone_onnx) | |

### The parameters of the constructor of MeloTTX_ONNX:
- **model_path** str. The path of the folder store the model.
- **execution_provider** str. The device for the onnxruntime, CUDA, CPU, or others. If it's None, the library will choose the better one between the CUDA and CPU.
- **verbose** bool. Set True for display the detail information when the library working.
- **onnx_session_options** onnxruntim.SessionOptions. You can setup the special options for the onnx session.
- **onnx_params** dict. The other parameters you want to pass into the onnxruntim.InferenceSession 

### The parameters of the MeloTTX_ONNX.speak:
- **text** str. The text you want to synthesis.
- **speaker** str. The speaker you want to use.
- **speed** float. The speed of the speech
- **sdp_ratio** float.
- **noise_scale** float.
- **noise_scale_w** float.
- **pbar** function. Such as tqdm
- **Returns** numpy.array. The data of the result audio

### Some useful property of the instance of the MeloTTS:
- **speakers**: [str]. Readonly. The available speakers
- **sample_rate**: int. Readonly. The sample rate of the synthesis result
- **language**: str. Readonly. The language of the current model

### The parameters of the constructor of OpenVoiceToneClone_ONNX:
- **model_path** str. The path of the folder store the model.
- **execution_provider** str. The device for the onnxruntime, CUDA, CPU, or others. If it's None, the library will choose the better one between the CUDA and CPU.
- **verbose** bool. Set True for display the detail information when the library working.
- **onnx_session_options** onnxruntim.SessionOptions. You can setup the special options for the onnx session.
- **onnx_params** dict. The other parameters you want to pass into the onnxruntim.InferenceSession 

### The parameters of OpenVoiceToneClone_ONNX.extract_tone_color
- **audio** numpy.array. The data of the audio
- **Returns** numpy.array. The tone color vector

### The parameters of OpenVoiceToneClone_ONNX.mix_tone_color
- **color** list[numpy.array]. The list of the tone colors you want to mix. Each element should be the result of extract_tone_color.
- **Returns** numpy.array. The tone color vector

### The parameters of OpenVoiceToneClone_ONNX.tone_clone
- **audio** numpy.array. The data of the audio that will be changed the tone
- **target_tone_color** numpy.array. The tone color that you want to clone. It should be the result of the extract_tone_color or mix_tone_color.
- **tau** float
- **Returns** numpy.array. The dest audio

### The parameters of OpenVoiceToneClone_ONNX.resample
- **audio** numpy.array. The source audio you want to resample.
- **original_rate** int. The original sample rate of the source audio
- **Returns** numpy.array. The dest data of the audio after resample
