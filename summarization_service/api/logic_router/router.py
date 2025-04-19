import asyncio
import os
import tempfile
from io import BytesIO

import torch
import torchaudio
import whisper
from fastapi import APIRouter, HTTPException, UploadFile, status
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast

from tools.separation import separate_sources

LogicRouter = APIRouter(tags=["Logic"])

CLEANER_MODEL = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
TRANS_MODEL = whisper.load_model("medium")
MODEL_NAME = "UrukHan/t5-russian-summarization"
TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_NAME)
SUMM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLEANER_MODEL.to(DEVICE)
TRANS_MODEL.to(DEVICE)
SUMM_MODEL.to(DEVICE)

MAX_INPUT = 4048
MAX_OUTPUT = 256
SUMM_MODEL.config.max_length = MAX_OUTPUT


@LogicRouter.post(
    "/process",
    response_description="Краткое содержание видеоконференции",
    description="Обрабатывает аудиодорожку видеофайла и оставляет из нее только нужную информацию в текстовом формате",
    summary="Обработка алгоритмом суммаризации",
)
async def process(file: UploadFile) -> str:
    if file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось найти имя файла",
        )
    suffix = file.filename[file.filename.rfind(".") :]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    process = await asyncio.create_subprocess_shell(
        f"ffmpeg -i {tmp_path} -vn -c:a pcm_s16le -f wav pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    audio_bytes, _ = await process.communicate()
    audio_buffer = BytesIO(audio_bytes)
    os.unlink(tmp_path)
    torch_audio, _sr = torchaudio.load(audio_buffer)
    torch_audio = torch_audio.to(DEVICE)

    if _sr != HDEMUCS_HIGH_MUSDB_PLUS.sample_rate:
        resampler = torchaudio.transforms.Resample(
            _sr,
            HDEMUCS_HIGH_MUSDB_PLUS.sample_rate,
        ).to(DEVICE)
        torch_audio = resampler(torch_audio)

    if process.returncode != 0 or torch_audio.numel() == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Файл не удалось обработать",
        )
    sources = separate_sources(
        CLEANER_MODEL,
        torch_audio,
        HDEMUCS_HIGH_MUSDB_PLUS.sample_rate,
    )

    mono_audio = sources["vocals"].mean(0)
    if _sr != 16000:
        resampler = torchaudio.transforms.Resample(_sr, 16000).to(DEVICE)
        mono_audio = resampler(mono_audio)
    text = TRANS_MODEL.transcribe(mono_audio, language="ru")["text"]

    if len(text) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Текст распознать не удалось",
        )

    input_data = TOKENIZER(
        [
            "Напиши основные идеи, решения и действия из этого текста. Дай результат в формате списка. : "
            + text,
        ],
        padding="longest",
        max_length=MAX_INPUT,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    predicts = SUMM_MODEL.generate(input_data.to(DEVICE))

    return TOKENIZER.batch_decode(predicts, skip_special_tokens=True)[0]
