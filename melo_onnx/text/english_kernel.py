import pickle
import os
import re
from g2p_en import G2p

from . import symbols

from .english_utils.abbreviations import expand_abbreviations
from .english_utils.time_norm import expand_time_english
from .english_utils.number_norm import normalize_numbers

from .japanese_kernel import distribute_phone

from .abstract_kernel import AbstractKernel

arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}

def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone

def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones

def post_replace_ph(ph):
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
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")

def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict

def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)

class EnglishKernel(AbstractKernel):
    def __init__(self, model_path:str, onnx_providers:list, session_opts = None, onnx_params:dict = None):
        super().__init__(model_path, onnx_providers, session_opts, onnx_params)

    __eng_dict = None

    __g2p = G2p()

    @staticmethod
    def get_eng_dict():
        if EnglishKernel.__eng_dict is None:
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "rb") as pickle_file:
                    g2p_dict = pickle.load(pickle_file)
            else:
                g2p_dict = read_dict()
                cache_dict(g2p_dict, CACHE_PATH)
            EnglishKernel.__eng_dict = g2p_dict
        return EnglishKernel.__eng_dict
    
    def text_normalize(self, text):
        text = text.lower()
        text = expand_time_english(text)
        text = normalize_numbers(text)
        text = expand_abbreviations(text)
        return text
    
    @staticmethod
    def g2p_en(text, pad_start_end = True, tokenized = None):
        if tokenized is None:
            raise ValueError("tokenized must not be none")
        phs = []
        ph_groups = []
        for t in tokenized:
            if not t.startswith("#"):
                ph_groups.append([t])
            else:
                ph_groups[-1].append(t.replace("#", ""))
        
        phones = []
        tones = []
        word2ph = []
        eng_dict = EnglishKernel.get_eng_dict()
        for group in ph_groups:
            w = "".join(group)
            phone_len = 0
            word_len = len(group)
            if w.upper() in eng_dict:
                phns, tns = refine_syllables(eng_dict[w.upper()])
                phones += phns
                tones += tns
                phone_len += len(phns)
            else:
                phone_list = list(filter(lambda p: p != " ", EnglishKernel.__g2p(w)))
                for ph in phone_list:
                    if ph in arpa:
                        ph, tn = refine_ph(ph)
                        phones.append(ph)
                        tones.append(tn)
                    else:
                        phones.append(ph)
                        tones.append(0)
                    phone_len += 1
            aaa = distribute_phone(phone_len, word_len)
            word2ph += aaa
        phones = [post_replace_ph(i) for i in phones]

        if pad_start_end:
            phones = ["_"] + phones + ["_"]
            tones = [0] + tones + [0]
            word2ph = [1] + word2ph + [1]
        return phones, tones, word2ph

    def g2p(self, text, **kwargs):
        pad_start_end = True if 'pad_start_end' not in kwargs else kwargs['pad_start_end']
        tokenized = self._tokenizer(text) if 'tokenized' not in kwargs else kwargs['tokenized']
        return EnglishKernel.g2p_en(text, pad_start_end, tokenized)
    