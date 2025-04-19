import asyncio
import os
import tempfile
from io import BytesIO

import librosa
import whisper
from fastapi import APIRouter, HTTPException, UploadFile, status
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast

LogicRouter = APIRouter(tags=["Logic"])

# Зададим название выбронной модели из хаба
MODEL_NAME = "UrukHan/t5-russian-summarization"

TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_NAME)
SUMM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
TRANS_MODEL = whisper.load_model("tiny")


@LogicRouter.post(
    "/process",
    response_description="Краткое содержание видеоконференции",
    description="Обрабатывает аудиодорожку видеофайла и оставляет из нее только нужную информацию в текстовом формате",
    summary="Обработка алгоритмом суммаризации",
)
async def process(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    process = await asyncio.create_subprocess_shell(
        f"ffmpeg -i {tmp_path} -vn -c:a pcm_s16le -f wav pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        # stderr=subprocess.PIPE,
    )
    audio_bytes, _ = await process.communicate()
    audio_buffer = BytesIO(audio_bytes)
    os.unlink(tmp_path)
    np_audio, _sr = librosa.load(audio_buffer, sr=None)

    if process.returncode != 0 or np_audio.size == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Файл не удалось обработать",
        )

    text = TRANS_MODEL.transcribe(np_audio, language="ru")["text"]

    if len(text) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Текст распознать не удалось",
        )

    input_data = TOKENIZER(
        ["Spell correct: " + sequence for sequence in text],
        padding="longest",
        return_tensors="pt",
    ).input_ids

    predicts = SUMM_MODEL.generate(input_data)

    return TOKENIZER.batch_decode(predicts, skip_special_tokens=True)[0]
