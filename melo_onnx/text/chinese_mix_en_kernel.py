import re

from .symbols import language_tone_start_map

from .chinese_kernel import ChineseKernel
from .english_kernel import EnglishKernel

class ChineseMixEnKernel(ChineseKernel):
    def __init__(self, model_path:str, onnx_providers:list, session_opts = None, onnx_params:dict = None):
        super().__init__(model_path, onnx_providers, session_opts, onnx_params)
    
    punctuation = ["!", "?", "…", ",", ".", "'", "-"]

    def _process_cv(self, c, v, seg):
        if c == 'EN_WORD':
            tokenized_en = self._tokenizer.encode(v, add_special_tokens=False)
            phones_en, tones_en, word2ph_en = EnglishKernel.g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
            # apply offset to tones_en
            tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
            return phones_en, tones_en, word2ph_en
        else:
            return super()._process_cv(c, v, seg)
        
    def _process_seg_cut_item(self, word, pos):
        if pos == "eng":
            return True, ['EN_WORD'], [word]
        return False, None, None
    
    def _process_seg(self, seg):
        return seg
    
    def _g2p(self, segments, impl='v2', **kwargs):
        if impl == 'v1':
            return super()._g2p(segments, **kwargs)
        elif impl == 'v2':
            return self._g2p_v2(segments)
        else:
            raise NotImplementedError(f"Unknown implementation {impl}")
        
    def replace_punctuation(self, text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        pattern = re.compile("|".join(re.escape(p) for p in self.rep_map.keys()))
        replaced_text = pattern.sub(lambda x: self.rep_map[x.group()], text)
        replaced_text = re.sub(r"[^\u4e00-\u9fa5_a-zA-Z\s" + "".join(self.punctuation) + r"]+", "", replaced_text)
        replaced_text = re.sub(r"[\s]+", " ", replaced_text)
        return replaced_text

    def _g2p_v2(self, segments):
        spliter = '#$&^!@'

        phones_list = []
        tones_list = []
        word2ph = []

        for text in segments:
            assert spliter not in text
            # replace all english words
            text = re.sub('([a-zA-Z\s]+)', lambda x: f'{spliter}{x.group(1)}{spliter}', text)
            texts = text.split(spliter)
            texts = [t for t in texts if len(t) > 0]
            
            for text in texts:
                if re.match('[a-zA-Z\s]+', text):
                    # english
                    tokenized_en = self._tokenizer.encode(text, add_special_tokens=False).tokens
                    phones_en, tones_en, word2ph_en = EnglishKernel.g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
                    # apply offset to tones_en
                    tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
                    phones_list += phones_en
                    tones_list += tones_en
                    word2ph += word2ph_en
                else:
                    phones_zh, tones_zh, word2ph_zh = super()._g2p([text])
                    phones_list += phones_zh
                    tones_list += tones_zh
                    word2ph += word2ph_zh
        return phones_list, tones_list, word2ph
    