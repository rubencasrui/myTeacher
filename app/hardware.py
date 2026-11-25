from dataclasses import dataclass
import os
import ctranslate2
import os
import site
from pathlib import Path


@dataclass
class HardwareProfile:
    name: str
    device: str
    compute_type: str
    default_model: str
    cpu_threads: int
    num_workers: int


def register_nvidia_dll_paths() -> None:
    if os.name != "nt":
        return
    candidate_roots: list[Path] = []
    for sp in site.getsitepackages():
        candidate_roots.append(Path(sp) / "nvidia")
    user_site = site.getusersitepackages()
    if user_site:
        candidate_roots.append(Path(user_site) / "nvidia")

    added: list[str] = []
    current_path = os.environ.get("PATH", "")
    for root in candidate_roots:
        if root.exists():
            for dll_dir in root.glob("*/bin"):
                if dll_dir.is_dir():
                    s_dir = str(dll_dir)
                    if s_dir not in current_path:
                        added.append(s_dir)

    if added:
        os.environ["PATH"] = os.pathsep.join(added + [current_path])
        if hasattr(os, "add_dll_directory"):
            for d in added:
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass

def transcribe_with_fallback(
    profile: HardwareProfile,
    models_dir: Path,
    ffmpeg_exe: Path,
    video_path: Path,
    srt_path: Path,
    json_path: Path,
    model_size: str | None = None,
    initial_prompt: str | None = None,
) -> list[dict]:
    try:
        model = load_whisper_model(profile, models_dir, model_size=model_size)
        return transcribe_video_to_files(
            model, ffmpeg_exe, video_path, srt_path, json_path, initial_prompt=initial_prompt
        )
    except RuntimeError as exc:
        if profile.device == "cuda" and ("cublas" in str(exc).lower() or "cudnn" in str(exc).lower() or "dll" in str(exc).lower()):
            logger.warning("Fallo cargando/ejecutando CUDA (%s). Cayendo automáticamente a CPU int8.", exc)
            cpu_profile = HardwareProfile(
                name="cpu_int8_fallback",
                device="cpu",
                compute_type="int8",
                default_model="small",
                cpu_threads=max(2, min(4, (os.cpu_count() or 4) // 2)),
                num_workers=1,
            )
            cpu_model = load_whisper_model(cpu_profile, models_dir, model_size="small")
            return transcribe_video_to_files(
                cpu_model, ffmpeg_exe, video_path, srt_path, json_path, initial_prompt=initial_prompt
            )
        raise


def detect_hardware_profile() -> HardwareProfile:
    """Detecta automáticamente GPU CUDA o cae a perfil CPU optimizado."""
    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_count = 0

    if cuda_count > 0:
        return HardwareProfile(
            name="gpu_cuda",
            device="cuda",
            compute_type="int8_float16",
            default_model="medium",
            cpu_threads=0,
            num_workers=1,
        )

    total_cpus = os.cpu_count() or 4
    safe_threads = max(2, min(4, total_cpus // 2))
    return HardwareProfile(
        name="cpu_int8",
        device="cpu",
        compute_type="int8",
        default_model="small",
        cpu_threads=safe_threads,
        num_workers=1,
    )