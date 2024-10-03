# MeloTTS by ONNX
[English](./README.md)

## 介绍
这是MeloTTS的ONNX运行时的实现，以用于在运行时加速整个TTS的过程。
目前在Intel Core i7 10代1.3GHz处理器上，测试“今天天气真nice”的合成速度约为0.95s。

我们重构了文本处理工具集，让模块的加载和运行可以尽可能提速。
我们将用到的模型转换成了onnx格式，并归并到同一目录下。模型文件目前以大文件形式存放在本仓库的models文件夹下面。使用时需要自行下载。

我们目前仅实现了zh-mix-en（即中英混合）语言。其他的语言支持在后续逐步补充。

## 使用方法

```python

from melo_onnx import MeloTTX_ONNX
import soundfile

model_path = "path/to/folder/of/model"
tts = MeloTTX_ONNX(model_path)
audio = tts.speak("今天天气真nice。", tts.speakers[0])

soundfile.write("path/of/audio.wav", audio, samplerate=tts.sample_rate)

```

MeloTTX_ONNX构造函数的参数如下：
- **model_path** str. 模型存放目录的路径.
- **execution_provider** str. onnxruntime使用的设备，比如CUDA、CPU、等等。如果这个参数传入None，系统会在CUDA和CPU中选择一个最佳的设备。
- **verbose** bool. 设置True表示在系统工作中输出详细的信息，一般可用来做调试。
- **onnx_session_options** onnxruntim.SessionOptions. 用来传入指定的onnx推理会话参数
- **onnx_params** dict. 你系统在onnxruntim.InferenceSession构造时传入的其他参数

MeloTTX_ONNX.speak方法的参数如下:
- **text** str. 需要进行合成的文本
- **speaker** str. 你想使用的发音者
- **speed** float. 语音的速度
- **sdp_ratio** float.
- **noise_scale** float.
- **noise_scale_w** float.
- **pbar** function. Such as tqdm

MeloTTS实例的可用属性:
- **speakers**: [str]. 只读。有效的发音者列表
- **sample_rate**: int. 只读。合成结果的采样率
- **language**: str. 只读。当前使用的模型的语言