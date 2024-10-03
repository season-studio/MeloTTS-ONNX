print("Import basic packages...")
import os
import sys
import onnxruntime as ort

from test_tools import *

sys.path.append(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))
sys.path.append(parent_dir)

verbose = '--verbose' in sys.argv

#from melo_onnx.text.chinese_mix_en_kernel import ChineseMixEnKernel

#c = ChineseMixEnKernel("./models/zh_mix_en")
#nt = c.text_normalize("今天的天气真nice")
#print(nt)
#print(c.g2p(nt))

onnx_params = {
    'provider_options': [{
        'device_type': 'GPU'
    }]
}

#session_opts = ort.SessionOptions()

print("Import MeloTTS_ONNX...")
from melo_onnx import MeloTTS_ONNX

print("Create instance of MeloTTS_ONNX...")
tts = MeloTTS_ONNX("./models/zh_mix_en", verbose=verbose, execution_provider="CPU", onnx_params=onnx_params)

print("Speak some sentences and save to test.wav...")
audio = tts.speak('今天的天气真nice。咱们一起去郊游吧。然后，咱们到hotel里面去开个party。', 'ZH')
print(audio)

import soundfile
soundfile.write("./test.wav", audio, samplerate=tts.sample_rate)


print("Measure time...")

@do_repeat(10)
@measure_time
def tts_time(**kwargs):
    return tts.speak('今天天气真nice', 'ZH')

print(f"Average time is {get_avg_value(tts_time)}s")