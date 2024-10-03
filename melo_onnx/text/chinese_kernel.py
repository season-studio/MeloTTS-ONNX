import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from .tone_sandhi import ToneSandhi

import jieba.posseg as psg

from .abstract_kernel import AbstractKernel

class ChineseKernel(AbstractKernel):
    def __init__(self, model_path:str, onnx_providers:list, session_opts = None, onnx_params:dict = None):
        super().__init__(model_path, onnx_providers, session_opts, onnx_params)

    pinyin_to_symbol_map = {
        line.split("\t")[0]: line.strip().split("\t")[1]
        for line in open(os.path.join(os.path.dirname(__file__), "opencpop-strict.txt")).readlines()
    }

    tone_modifier = ToneSandhi()

    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "$": ".",
        "“": "'",
        "”": "'",
        "‘": "'",
        "’": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
    }

    punctuation = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡"]

    def replace_punctuation(self, text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        pattern = re.compile("|".join(re.escape(p) for p in self.rep_map.keys()))

        replaced_text = pattern.sub(lambda x: self.rep_map[x.group()], text)

        replaced_text = re.sub(
            r"[^\u4e00-\u9fa5" + "".join(self.punctuation) + r"]+", "", replaced_text
        )

        return replaced_text
    
    def g2p(self, text, **kwargs):
        pattern = r"(?<=[{0}])\s*".format("".join(self.punctuation))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        phones, tones, word2ph = self._g2p(sentences, **kwargs)
        assert sum(word2ph) == len(phones)
        # assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
        return phones, tones, word2ph
    
    def _get_initials_finals(self, word):
        initials = []
        finals = []
        orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
        orig_finals = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
        )
        for c, v in zip(orig_initials, orig_finals):
            initials.append(c)
            finals.append(v)
        return initials, finals
    
    def _process_cv(self, c, v, seg):
        raw_pinyin = c + v
        # NOTE: post process for pypinyin outputs
        # we discriminate i, ii and iii
        if c == v:
            assert c in self.punctuation
            phone = [c]
            return phone, [0] * len(phone), [1]
        else:
            v_without_tone = v[:-1]
            tone = v[-1]

            pinyin = c + v_without_tone
            assert tone in "12345"

            if c:
                # 多音节
                v_rep_map = {
                    "uei": "ui",
                    "iou": "iu",
                    "uen": "un",
                }
                if v_without_tone in v_rep_map.keys():
                    pinyin = c + v_rep_map[v_without_tone]
            else:
                # 单音节
                pinyin_rep_map = {
                    "ing": "ying",
                    "i": "yi",
                    "in": "yin",
                    "u": "wu",
                }
                if pinyin in pinyin_rep_map.keys():
                    pinyin = pinyin_rep_map[pinyin]
                else:
                    single_rep_map = {
                        "v": "yu",
                        "e": "e",
                        "i": "y",
                        "u": "w",
                    }
                    if pinyin[0] in single_rep_map.keys():
                        pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

            assert pinyin in self.pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
            phone = self.pinyin_to_symbol_map[pinyin].split(" ")
            len_phone = len(phone)
            
            return phone, [int(tone)] * len_phone, [len_phone]
        
    def _process_seg_cut_item(self, word, pos):
        if pos == "eng":
            return True, None, None
        return False, None, None
    
    def _process_seg(self, seg):
        return re.sub("[a-zA-Z]+", "", seg)
    
    def _g2p(self, segments, **kwargs):
        phones_list = []
        tones_list = []
        word2ph = []
        for seg in segments:
            # Replace all English words in the sentence
            seg = self._process_seg(seg)
            seg_cut = psg.lcut(seg)
            initials = []
            finals = []
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
            for word, pos in seg_cut:
                _ret, _init, _final = self._process_seg_cut_item(word, pos)
                if _ret:
                    if _init is not None:
                        initials.append(_init)
                    if _final is not None:
                        finals.append(_final)
                else:
                    sub_initials, sub_finals = self._get_initials_finals(word)
                    sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                    initials.append(sub_initials)
                    finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])
            #
            for c, v in zip(initials, finals):
                phone, tone, w2p = self._process_cv(c, v, seg)
                word2ph += w2p
                phones_list += phone
                tones_list += tone

        return phones_list, tones_list, word2ph
    
    def text_normalize(self, text):
        numbers = re.findall(r"\d+(?:\.?\d+)?", text)
        for number in numbers:
            text = text.replace(number, cn2an.an2cn(number), 1)
        text = self.replace_punctuation(text)
        return text
    