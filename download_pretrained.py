from espnet_model_zoo.downloader import ModelDownloader
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import time
import torch
from scipy.io.wavfile import write
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model


def download_pretrained(lang):
    if lang == 'en':
        tag = 'kan-bayashi/ljspeech_conformer_fastspeech2' #@param ["kan-bayashi/ljspeech_tacotron2", "kan-bayashi/ljspeech_fastspeech", "kan-bayashi/ljspeech_fastspeech2", "kan-bayashi/ljspeech_conformer_fastspeech2", "kan-bayashi/ljspeech_vits"] {type:"string"}
        vocoder_tag = "ljspeech_hifigan.v1" #@param ["none", "parallel_wavegan/ljspeech_parallel_wavegan.v1", "parallel_wavegan/ljspeech_full_band_melgan.v2", "parallel_wavegan/ljspeech_multi_band_melgan.v2", "parallel_wavegan/ljspeech_hifigan.v1", "parallel_wavegan/ljspeech_style_melgan.v1"] {type:"string"}
    elif lang == 'ch':
        tag = 'kan-bayashi/csmsc_fastspeech2' #@param ["kan-bayashi/csmsc_tacotron2", "kan-bayashi/csmsc_transformer", "kan-bayashi/csmsc_fastspeech", "kan-bayashi/csmsc_fastspeech2", "kan-bayashi/csmsc_conformer_fastspeech2", "kan-bayashi/csmsc_full_band_vits"] {type: "string"}
        vocoder_tag = "csmsc_hifigan.v1" #@param ["none", "parallel_wavegan/csmsc_parallel_wavegan.v1", "parallel_wavegan/csmsc_multi_band_melgan.v2", "parallel_wavegan/csmsc_hifigan.v1", "parallel_wavegan/csmsc_style_melgan.v1"] {type:"string"}
    elif lang == 'ja':
        tag = "kan-bayashi/jvs_jvs010_vits_accent_with_pause"  # @param ["kan-bayashi/jsut_tacotron2", "kan-bayashi/jsut_transformer", "kan-bayashi/jsut_fastspeech", "kan-bayashi/jsut_fastspeech2", "kan-bayashi/jsut_conformer_fastspeech2", "kan-bayashi/jsut_conformer_fastspeech2_accent", "kan-bayashi/jsut_conformer_fastspeech2_accent_with_pause", "kan-bayashi/jsut_vits_accent_with_pause", "kan-bayashi/jsut_full_band_vits_accent_with_pause", "kan-bayashi/jvs_jvs001_vits_accent_with_pause", "kan-bayashi/jvs_jvs010_vits_accent_with_pause"] {type:"string"}
        vocoder_tag = "jsut_hifigan.v1"

    d = ModelDownloader('./pretrained_model/' + lang)
    d.download_and_unpack(tag)
    download_pretrained_model(vocoder_tag, './pretrained_model/' + lang)


download_pretrained('en')
