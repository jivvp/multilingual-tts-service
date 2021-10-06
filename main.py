# -*- coding: utf-8 -*-
import torch
import argparse
import numpy as np
import librosa
import hashlib
import struct
import time
import re
import os
import base64
import setproctitle

import uvicorn
from typing import Dict
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from atomic_counter import AtomicCounter

import sys

from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import time
import torch
from scipy.io.wavfile import write


class StatusOutput(BaseModel):
    status: str
    totalRequest: int
    waitRequest: int


tag_metadata = [{"name": "synthesis"}]

app = FastAPI(title="Smartsinage MultilingualTTS", version="0.0.1", openapi_tags=tag_metadata)

loaded_model = None
loaded_vocoder = None
loaded_denoiser = None
executor = None
korean_cleaner = None
n_workers = 0
counter = AtomicCounter()


@app.get("/status", response_model=StatusOutput, tags=["synthesis"])
def server_check():
    n_request = counter.get()
    status = "running" if n_request > 0 else "wait"
    wait_request = n_request - n_workers if n_request > n_workers else 0
    return {"status": status, "totalRequest": n_request, "waitRequest": wait_request}


@app.get("/tts/stream", tags=["synthesis"])
def test(text: str):
    print('***********************')
    task = partial(_infer, text)
    counter.increase()
    future = executor.submit(task)
    text_to_wav, duration = future.result()
    counter.decrease()

    return Response(
        content=text_to_wav,
        headers={"Audio-Duration": str(round(duration, 2))},
        media_type="audio/wav",
    )


def _load_model(lang):
    print('load_model start')
    if lang =='en':
        train_config = './pretrained_model/en/a8ad80d053f86e46f40af56430f0db22/exp/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/config.yaml'
        model_file = './pretrained_model/en/a8ad80d053f86e46f40af56430f0db22/exp/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/train.loss.ave_5best.pth'
        vocoder_file = './pretrained_model/en/ljspeech_hifigan.v1/checkpoint-2500000steps.pkl'
        vocoder_config = './pretrained_model/en/ljspeech_hifigan.v1/config.yml'

    elif lang == 'ja':
        train_config = './pretrained_model/ja/0484862b63452fb1369e0e0a2ac7df98/exp/tts_finetune_jvs010_jsut_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/config.yaml'
        model_file = './pretrained_model/ja/0484862b63452fb1369e0e0a2ac7df98/exp/tts_finetune_jvs010_jsut_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/50epoch.pth'
        vocoder_file = './pretrained_model/ja/jsut_hifigan.v1/checkpoint-2500000steps.pkl'
        vocoder_config = './pretrained_model/ja/jsut_hifigan.v1/config.yml'

    elif lang == 'ch':
        train_config = './pretrained_model/ch/cc283e28cc6cec58ef92ab4d74b476f4/exp/tts_train_fastspeech2_raw_phn_pypinyin_g2p_phone/config.yaml'
        model_file ='./pretrained_model/ch/cc283e28cc6cec58ef92ab4d74b476f4/exp/tts_train_fastspeech2_raw_phn_pypinyin_g2p_phone/train.loss.ave_5best.pth'
        vocoder_file = './pretrained_model/ch/csmsc_hifigan.v1/checkpoint-2500000steps.pkl'
        vocoder_config = './pretrained_model/ch/csmsc_hifigan.v1/config.yml'

    text2speech = Text2Speech.from_pretrained(
        train_config = train_config,
        model_file = model_file,
        vocoder_file = vocoder_file,
        vocoder_config = vocoder_config,
        device = "cuda",
        threshold=0.5,
        # Only for Tacotron 2
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2 & VITS
        speed_control_alpha=1.0,
        # Only for VITS
        noise_scale=0.667,
        noise_scale_dur=0.8,
    )
    print('load model')
    return text2speech


def _infer(text):
    print("start!")
    print(text)

    with torch.no_grad():
        start_time = time.time()
        wav = text2speech(text)["wav"]
        wav = wav.squeeze()
        wav = wav.cpu().detach().numpy().astype('float32')

    audio_duration = librosa.get_duration(wav, 22050)
    wav *= 32768 / max(0.01, np.max(np.abs(wav)))
    wav = wav.astype(np.int16)
    
    wav = _convert_to_pcm16(wav, 22050)
    print("{} seconds".format(time.time() - start_time))

    return wav, audio_duration


def _convert_to_pcm16(wav_int16, sr):
    data = wav_int16
    dkind = data.dtype.kind
    fs = sr

    header_data = b""
    header_data += b"RIFF"
    header_data += b"\x00\x00\x00\x00"
    header_data += b"WAVE"
    header_data += b"fmt "

    format_tag = 0x0001  # WAVE_FORMAT_PCM
    channels = 1
    bit_depth = data.dtype.itemsize * 8
    bytes_per_second = fs * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)

    fmt_chunk_data = struct.pack(
        "<HHIIHH", format_tag, channels, fs, bytes_per_second, block_align, bit_depth
    )
    header_data += struct.pack("<I", len(fmt_chunk_data))
    header_data += fmt_chunk_data

    if ((len(header_data) - 4 - 4) + (4 + 4 + data.nbytes)) > 0xFFFFFFFF:
        raise ValueError("Data exceeds wave file size limit")

    data_chunk_data = b"data"
    data_chunk_data += struct.pack("<I", data.nbytes)
    header_data += data_chunk_data

    data_bytes = data.tobytes()

    data_pcm16 = header_data
    data_pcm16 += data_bytes

    return data_pcm16


def setOutput(result, errorCode=0, errorMessage=""):
    return {"result": result, "errorCode": errorCode, "errorMessage": errorMessage}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)

    parser.add_argument(
        "--lang", type=str, default="en"
    )

    args = parser.parse_args()
    n_workers = 8
    proc_name = "smartsinage_{}".format(args.lang)
    setproctitle.setproctitle(proc_name)

    text2speech = _load_model(args.lang)

    executor = ThreadPoolExecutor(max_workers=n_workers)

    print("start proces....")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
