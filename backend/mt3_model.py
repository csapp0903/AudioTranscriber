"""
MT3 (Music Transcription with Transformers) 模型管理模块

此模块负责:
1. 自动下载和缓存 MT3 预训练模型
2. 模型加载和初始化
3. 音频预处理
4. 推理和后处理

参考文档:
- MT3 官方仓库: https://github.com/magenta/mt3
- T5X 框架: https://github.com/google-research/t5x
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import threading

import numpy as np

# ==============================================================================
# 配置 JAX 后端 (在导入 JAX 之前必须设置)
# ==============================================================================
# 如果没有 GPU，强制使用 CPU
# 可以通过环境变量 JAX_PLATFORM_NAME 覆盖
if os.environ.get("JAX_PLATFORM_NAME") is None:
    # 检测是否有 GPU 可用
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            logging.info("No GPU detected, using CPU backend for JAX")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        logging.info("nvidia-smi not found, using CPU backend for JAX")

# 设置 JAX 内存配置 (防止 OOM)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

# ==============================================================================
# 延迟导入重型库 (仅在需要时加载)
# ==============================================================================

logger = logging.getLogger(__name__)

# 模型单例
_model_instance = None
_model_lock = threading.Lock()


class MT3ModelManager:
    """
    MT3 模型管理器

    使用单例模式确保模型只加载一次，避免内存浪费。
    """

    # MT3 模型配置
    SAMPLE_RATE = 16000  # MT3 需要 16kHz 采样率
    SPECTROGRAM_CONFIG = {
        "sample_rate": 16000,
        "hop_width": 128,
        "num_mel_bins": 512,
    }

    # 模型路径配置
    DEFAULT_CHECKPOINT_DIR = os.environ.get(
        "MT3_CHECKPOINT_DIR",
        "/app/models/mt3"
    )
    GCS_CHECKPOINT_PATH = "gs://mt3/checkpoints/mt3/"

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        初始化 MT3 模型管理器

        Args:
            checkpoint_dir: 模型 checkpoint 目录路径
        """
        self.checkpoint_dir = checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR
        self._model = None
        self._inference_model = None
        self._vocab = None
        self._loaded = False

        # 延迟导入
        self._imports_ready = False

    def _lazy_import(self):
        """延迟导入重型库"""
        if self._imports_ready:
            return

        logger.info("Importing MT3 dependencies (this may take a moment)...")

        # 导入所需模块
        global jax, jnp, t5x, mt3, note_seq, librosa, sf

        import jax
        import jax.numpy as jnp

        # 验证 JAX 后端
        devices = jax.devices()
        logger.info(f"JAX devices: {devices}")
        logger.info(f"JAX backend: {jax.default_backend()}")

        import librosa
        import soundfile as sf
        import note_seq

        # MT3 相关导入
        from mt3 import network
        from mt3 import note_sequences
        from mt3 import spectrograms
        from mt3 import vocabularies
        from mt3 import metrics_utils
        from mt3 import inference

        # 保存模块引用
        self._mt3_network = network
        self._mt3_note_sequences = note_sequences
        self._mt3_spectrograms = spectrograms
        self._mt3_vocabularies = vocabularies
        self._mt3_metrics_utils = metrics_utils
        self._mt3_inference = inference
        self._librosa = librosa
        self._sf = sf
        self._note_seq = note_seq
        self._jax = jax
        self._jnp = jnp

        self._imports_ready = True
        logger.info("MT3 dependencies imported successfully")

    def _ensure_checkpoint(self) -> bool:
        """
        确保 checkpoint 已下载

        Returns:
            bool: 是否成功获取 checkpoint
        """
        checkpoint_path = Path(self.checkpoint_dir)
        checkpoint_file = checkpoint_path / "checkpoint"

        if checkpoint_file.exists():
            logger.info(f"Checkpoint found at {self.checkpoint_dir}")
            return True

        logger.info("Checkpoint not found, attempting to download...")
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        try:
            # 使用 gsutil 下载 checkpoint
            result = subprocess.run(
                [
                    "gsutil", "-m", "cp", "-r",
                    f"{self.GCS_CHECKPOINT_PATH}*",
                    str(checkpoint_path)
                ],
                capture_output=True,
                text=True,
                timeout=600  # 10 分钟超时
            )

            if result.returncode == 0:
                logger.info("Checkpoint downloaded successfully")
                return True
            else:
                logger.error(f"Failed to download checkpoint: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Checkpoint download timed out")
            return False
        except FileNotFoundError:
            logger.error("gsutil not found. Please install Google Cloud SDK")
            return False

    def load_model(self) -> bool:
        """
        加载 MT3 模型

        Returns:
            bool: 是否成功加载模型
        """
        if self._loaded:
            logger.info("Model already loaded")
            return True

        try:
            # 延迟导入
            self._lazy_import()

            # 确保 checkpoint 存在
            if not self._ensure_checkpoint():
                raise RuntimeError("Failed to obtain MT3 checkpoint")

            logger.info("Loading MT3 model...")

            # 加载词汇表
            self._vocab = self._mt3_vocabularies.build_codec_vocabulary()

            # 创建模型配置
            # MT3 使用 T5 1.1 架构
            model_config = self._mt3_network.T5Config(
                vocab_size=self._vocab.vocab_size,
                dtype="bfloat16" if self._jax.default_backend() != "cpu" else "float32",
                emb_dim=512,
                num_heads=6,
                num_encoder_layers=8,
                num_decoder_layers=8,
                head_dim=64,
                mlp_dim=1024,
                mlp_activations=("gelu", "linear"),
                dropout_rate=0.0,
            )

            # 创建推理模型
            self._inference_model = self._mt3_inference.InferenceModel(
                checkpoint_path=self.checkpoint_dir,
                model_config=model_config,
                vocab=self._vocab,
            )

            self._loaded = True
            logger.info("MT3 model loaded successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to load MT3 model: {e}")
            return False

    def preprocess_audio(
        self,
        audio_path: str
    ) -> Tuple[np.ndarray, int]:
        """
        预处理音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        self._lazy_import()

        logger.info(f"Preprocessing audio: {audio_path}")

        # 加载音频并重采样到 16kHz
        audio, sr = self._librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 归一化
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()

        logger.info(f"Audio loaded: {len(audio) / sr:.2f} seconds at {sr}Hz")

        return audio, sr

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE
    ):
        """
        将音频转录为音符序列

        Args:
            audio: 音频数据 (numpy array)
            sample_rate: 采样率 (默认 16000)

        Returns:
            note_seq.NoteSequence: 转录的音符序列
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Starting transcription...")

        # 确保采样率正确
        if sample_rate != self.SAMPLE_RATE:
            logger.warning(
                f"Resampling from {sample_rate}Hz to {self.SAMPLE_RATE}Hz"
            )
            audio = self._librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.SAMPLE_RATE
            )

        # 执行推理
        # MT3 使用 spectrogram 作为输入
        est_ns = self._inference_model(audio)

        logger.info(
            f"Transcription complete: {len(est_ns.notes)} notes detected"
        )

        return est_ns

    def note_sequence_to_midi(
        self,
        note_sequence,
        output_path: str
    ) -> str:
        """
        将 NoteSequence 转换为 MIDI 文件

        Args:
            note_sequence: note_seq.NoteSequence 对象
            output_path: 输出 MIDI 文件路径

        Returns:
            str: 输出文件路径
        """
        self._lazy_import()

        logger.info(f"Converting to MIDI: {output_path}")

        # 使用 note_seq 库转换
        self._note_seq.sequence_proto_to_midi_file(
            note_sequence,
            output_path
        )

        logger.info(f"MIDI file saved: {output_path}")
        return output_path

    def process_audio_to_midi(
        self,
        input_audio_path: str,
        output_midi_path: str
    ) -> str:
        """
        完整的音频到 MIDI 转换流程

        Args:
            input_audio_path: 输入音频路径
            output_midi_path: 输出 MIDI 路径

        Returns:
            str: 输出 MIDI 文件路径
        """
        # 预处理
        audio, sr = self.preprocess_audio(input_audio_path)

        # 转录
        note_sequence = self.transcribe(audio, sr)

        # 保存 MIDI
        return self.note_sequence_to_midi(note_sequence, output_midi_path)


def get_model() -> MT3ModelManager:
    """
    获取 MT3 模型单例

    使用单例模式确保模型只加载一次。
    线程安全。

    Returns:
        MT3ModelManager: 模型管理器实例
    """
    global _model_instance

    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = MT3ModelManager()

    return _model_instance


def preload_model():
    """
    预加载模型 (用于 Worker 启动时)

    在 Celery Worker 启动时调用此函数可以预热模型，
    避免首次请求时的延迟。
    """
    logger.info("Preloading MT3 model...")
    model = get_model()
    success = model.load_model()
    if success:
        logger.info("MT3 model preloaded successfully")
    else:
        logger.error("Failed to preload MT3 model")
    return success
