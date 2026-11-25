import json
import logging
import subprocess
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel

from app.hardware import HardwareProfile

logger = logging.getLogger("myTeacher")


def format_srt_time(seconds: float) -> str:
    """Convierte segundos a formato SRT HH:MM:SS,mmm."""
    ms_total = max(0, int(round(seconds * 1000)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1_000
    millis = rem % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def extract_wav_16k_mono(ffmpeg_exe: Path, video_path: Path, wav_path: Path) -> None:
    """Extrae audio WAV 16kHz mono sin abrir ventana de consola en Windows."""
    if not ffmpeg_exe.exists():
        raise FileNotFoundError(f"No se encontró ffmpeg en: {ffmpeg_exe}")

    cmd = [
        str(ffmpeg_exe),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def load_whisper_model(profile: HardwareProfile, models_dir: Path, model_size: str | None = None) -> WhisperModel:
    """Carga el modelo faster-whisper según el perfil de hardware detectado."""
    chosen_model = model_size or profile.default_model
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Cargando Whisper '%s' en device=%s compute_type=%s threads=%s",
        chosen_model,
        profile.device,
        profile.compute_type,
        profile.cpu_threads,
    )
    return WhisperModel(
        chosen_model,
        device=profile.device,
        compute_type=profile.compute_type,
        cpu_threads=profile.cpu_threads,
        num_workers=profile.num_workers,
        download_root=str(models_dir),
    )


def transcribe_video_to_files(
    model: WhisperModel,
    ffmpeg_exe: Path,
    video_path: Path,
    srt_path: Path,
    json_path: Path,
    initial_prompt: str | None = None,
) -> list[dict]:
    """Transcribe un vídeo de clase en español y escribe .srt y .json en UTF-8."""
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="myteacher_audio_") as tmp_dir:
        wav_file = Path(tmp_dir) / "audio_16k.wav"
        extract_wav_16k_mono(ffmpeg_exe, video_path, wav_file)

        segments_gen, _ = model.transcribe(
            str(wav_file),
            language="es",
            beam_size=5,
            vad_filter=True,
            initial_prompt=initial_prompt,
        )

        segments_list: list[dict] = []
        srt_lines: list[str] = []

        for idx, seg in enumerate(segments_gen, start=1):
            text = (seg.text or "").strip()
            if not text:
                continue
            item = {
                "index": idx,
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": text,
            }
            segments_list.append(item)
            srt_lines.append(
                f"{idx}\n{format_srt_time(item['start'])} --> {format_srt_time(item['end'])}\n{text}\n"
            )

    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(segments_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return segments_list