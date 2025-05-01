import asyncio
import os
import tempfile
from io import BytesIO

import torch
import torchaudio
import whisper
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, status
from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

from tools.separation import separate_sources

load_dotenv()
if "GIGACHAT_CREDENTIALS" not in os.environ:
    msg = "Не обнаружен ключ авторизации в переменных среды для GigaChat"
    raise ValueError(msg)

LogicRouter = APIRouter(tags=["Logic"])

CLEANER_MODEL = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
TRANS_MODEL = whisper.load_model("medium")
SUMM_MODEL = GigaChat(
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    verify_ssl_certs=False,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLEANER_MODEL.to(DEVICE)
TRANS_MODEL.to(DEVICE)


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

    input_text = (
        "Напиши основные идеи, решения и действия из этого текста. Дай результат в формате списка. : "
        + text
    )

    return str(SUMM_MODEL.invoke([HumanMessage(content=input_text)]).content)
