"""
Celery 任务定义 - MT3 音乐转录

此模块定义了使用 MT3 (Music Transcription with Transformers) 进行音频转录的 Celery 任务。

流程:
1. 接收音频文件路径
2. (可选) 使用 Demucs 分离音轨
3. 使用 MT3 将音频转录为 MIDI
4. 使用 MuseScore 将 MIDI 转换为 PDF 乐谱

环境变量:
- MT3_CHECKPOINT_DIR: MT3 模型 checkpoint 目录
- JAX_PLATFORM_NAME: JAX 后端 ("cpu" 或 "gpu")
- ENABLE_DEMUCS: 是否启用 Demucs 音轨分离 ("true"/"false")
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from celery import signals

from backend.celery_app import celery_app
from backend.config import OUTPUT_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Worker 启动时预加载模型
# ==============================================================================
@signals.worker_process_init.connect
def init_worker(**kwargs):
    """
    Worker 进程初始化时预加载模型

    这确保模型在首次任务执行前就已加载到内存中，
    避免首次请求时的长延迟。
    """
    logger.info("Worker process initializing...")

    # 预加载 MT3 模型
    try:
        from backend.mt3_model import preload_model
        preload_model()
    except Exception as e:
        logger.error(f"Failed to preload MT3 model: {e}")
        # 不抛出异常，允许 Worker 启动
        # 模型会在首次使用时重试加载

    logger.info("Worker process initialization complete")


# ==============================================================================
# 音频预处理工具
# ==============================================================================
def ensure_compatible_audio(input_path: str, temp_dir: str) -> str:
    """
    确保音频格式与 MT3 兼容

    MT3 期望:
    - 采样率: 16kHz
    - 格式: WAV 或其他 librosa 支持的格式
    - 声道: 单声道

    如果输入不兼容，使用 ffmpeg 进行转换。

    Args:
        input_path: 输入音频路径
        temp_dir: 临时文件目录

    Returns:
        str: 兼容的音频文件路径
    """
    import subprocess

    input_path = Path(input_path)

    # 如果是 WAV 文件，直接返回 (librosa 会处理采样率)
    if input_path.suffix.lower() == ".wav":
        return str(input_path)

    # 转换为 WAV
    output_path = Path(temp_dir) / f"{input_path.stem}_converted.wav"

    logger.info(f"Converting audio to WAV: {input_path} -> {output_path}")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_path),
                "-ar", "16000",  # 16kHz 采样率
                "-ac", "1",  # 单声道
                "-y",  # 覆盖已存在的文件
                str(output_path)
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")

        logger.info("Audio conversion successful")
        return str(output_path)

    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        raise


# ==============================================================================
# Demucs 音轨分离 (可选)
# ==============================================================================
def separate_with_demucs(
    input_path: str,
    output_dir: str,
    model: str = "htdemucs"
) -> Optional[Dict[str, str]]:
    """
    使用 Demucs 分离音轨

    Args:
        input_path: 输入音频路径
        output_dir: 输出目录
        model: Demucs 模型名称

    Returns:
        Optional[Dict[str, str]]: 分离后的音轨路径，失败返回 None
    """
    import subprocess

    logger.info(f"Separating audio with Demucs: {input_path}")

    try:
        result = subprocess.run(
            [
                "python", "-m", "demucs",
                "--two-stems", "vocals",  # 只分离人声和伴奏
                "-n", model,
                "-o", output_dir,
                input_path
            ],
            capture_output=True,
            text=True,
            timeout=1800  # 30 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"Demucs separation failed: {result.stderr}")
            return None

        # 查找输出文件
        input_stem = Path(input_path).stem
        demucs_output = Path(output_dir) / model / input_stem

        if demucs_output.exists():
            tracks = {}
            for track_file in demucs_output.glob("*.wav"):
                track_name = track_file.stem
                tracks[track_name] = str(track_file)

            logger.info(f"Demucs separation complete: {tracks.keys()}")
            return tracks

        return None

    except FileNotFoundError:
        logger.warning("Demucs not installed. Skipping separation.")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Demucs separation timed out")
        return None


# ==============================================================================
# 主要任务: 使用 MT3 处理音频
# ==============================================================================
@celery_app.task(bind=True)
def process_audio(self, task_id: str, input_file: str) -> Dict[str, Any]:
    """
    使用 MT3 处理音频并生成 MIDI 和 PDF

    这是主要的 Celery 任务，执行完整的音乐转录流程。

    Args:
        task_id: 任务 ID
        input_file: 输入音频文件路径

    Returns:
        Dict[str, Any]: 处理结果
    """
    logger.info(f"Starting audio processing: task_id={task_id}, file={input_file}")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="mt3_")

    try:
        # 更新状态: 处理中
        self.update_state(
            state="PROCESSING",
            meta={"progress": 0, "stage": "preparing"}
        )

        # 创建输出目录
        task_output_dir = Path(OUTPUT_DIR) / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # 获取基础文件名
        base_name = Path(input_file).stem

        # 输出文件路径
        midi_path = str(task_output_dir / f"{base_name}.mid")
        pdf_path = str(task_output_dir / f"{base_name}.pdf")

        # ==================================================================
        # Step 1: 音频预处理
        # ==================================================================
        self.update_state(
            state="PROCESSING",
            meta={"progress": 10, "stage": "preprocessing"}
        )

        processed_audio = ensure_compatible_audio(input_file, temp_dir)

        # ==================================================================
        # Step 2: (可选) Demucs 音轨分离
        # ==================================================================
        enable_demucs = os.environ.get("ENABLE_DEMUCS", "false").lower() == "true"
        audio_to_transcribe = processed_audio

        if enable_demucs:
            self.update_state(
                state="PROCESSING",
                meta={"progress": 20, "stage": "separating"}
            )

            separated = separate_with_demucs(processed_audio, temp_dir)

            if separated and "no_vocals" in separated:
                # 使用伴奏轨道 (去掉人声更容易转录)
                audio_to_transcribe = separated["no_vocals"]
                logger.info("Using separated instrumental track")
            elif separated and "other" in separated:
                audio_to_transcribe = separated["other"]

        # ==================================================================
        # Step 3: MT3 转录
        # ==================================================================
        self.update_state(
            state="PROCESSING",
            meta={"progress": 40, "stage": "transcribing"}
        )

        logger.info("Starting MT3 transcription...")

        from backend.mt3_model import get_model

        model = get_model()

        # 确保模型已加载
        if not model.load_model():
            raise RuntimeError("Failed to load MT3 model")

        # 执行转录
        model.process_audio_to_midi(audio_to_transcribe, midi_path)

        logger.info(f"MIDI file created: {midi_path}")

        # ==================================================================
        # Step 4: 生成 PDF 乐谱
        # ==================================================================
        self.update_state(
            state="PROCESSING",
            meta={"progress": 80, "stage": "generating_pdf"}
        )

        logger.info("Generating PDF score...")

        from backend.musescore_utils import midi_to_pdf

        pdf_success = midi_to_pdf(midi_path, pdf_path)

        if not pdf_success:
            logger.warning("PDF generation failed, but MIDI is available")
            # 创建空的 PDF 占位符 (表示生成失败)
            # 用户仍可下载 MIDI 文件
            Path(pdf_path).touch()

        # ==================================================================
        # 完成
        # ==================================================================
        self.update_state(
            state="PROCESSING",
            meta={"progress": 100, "stage": "complete"}
        )

        logger.info(f"Processing complete: task_id={task_id}")

        return {
            "status": "SUCCESS",
            "task_id": task_id,
            "files": {
                "midi": midi_path,
                "pdf": pdf_path
            }
        }

    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "stage": "error"}
        )
        raise

    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ==============================================================================
# 辅助任务: 仅进行 MIDI 到 PDF 转换
# ==============================================================================
@celery_app.task(bind=True)
def convert_midi_to_pdf(self, task_id: str, midi_file: str) -> Dict[str, Any]:
    """
    将现有 MIDI 文件转换为 PDF 乐谱

    Args:
        task_id: 任务 ID
        midi_file: MIDI 文件路径

    Returns:
        Dict[str, Any]: 处理结果
    """
    logger.info(f"Converting MIDI to PDF: task_id={task_id}, file={midi_file}")

    try:
        self.update_state(state="PROCESSING", meta={"progress": 0})

        # 创建输出目录
        task_output_dir = Path(OUTPUT_DIR) / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # 输出文件路径
        base_name = Path(midi_file).stem
        pdf_path = str(task_output_dir / f"{base_name}.pdf")

        # 转换
        from backend.musescore_utils import midi_to_pdf

        if midi_to_pdf(midi_file, pdf_path):
            return {
                "status": "SUCCESS",
                "task_id": task_id,
                "files": {"pdf": pdf_path}
            }
        else:
            raise RuntimeError("PDF conversion failed")

    except Exception as e:
        logger.exception(f"Conversion failed: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


# ==============================================================================
# 健康检查任务
# ==============================================================================
@celery_app.task
def health_check() -> Dict[str, Any]:
    """
    检查 Worker 健康状态和模型加载情况

    Returns:
        Dict[str, Any]: 健康状态信息
    """
    import platform

    status = {
        "healthy": True,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # 检查 MT3 模型
    try:
        from backend.mt3_model import get_model
        model = get_model()
        status["mt3_loaded"] = model._loaded
        status["mt3_checkpoint_dir"] = model.checkpoint_dir
    except Exception as e:
        status["mt3_loaded"] = False
        status["mt3_error"] = str(e)

    # 检查 MuseScore
    try:
        from backend.musescore_utils import get_converter
        converter = get_converter()
        status["musescore_available"] = converter.is_available()
        status["musescore_path"] = converter.musescore_path
    except Exception as e:
        status["musescore_available"] = False
        status["musescore_error"] = str(e)

    # 检查 JAX 后端
    try:
        import jax
        status["jax_backend"] = jax.default_backend()
        status["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as e:
        status["jax_error"] = str(e)

    return status
