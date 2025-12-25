"""
MuseScore 命令行工具封装

此模块提供 MIDI 到 PDF/MusicXML 的转换功能。

依赖:
- musescore3 (系统包)
- xvfb (虚拟显示器，用于无头运行)
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class MuseScoreConverter:
    """
    MuseScore 转换器

    将 MIDI 文件转换为乐谱格式 (PDF, MusicXML 等)
    """

    # MuseScore 可执行文件路径
    MUSESCORE_PATHS = [
        "musescore3",
        "mscore3",
        "musescore",
        "/usr/bin/musescore3",
        "/usr/bin/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    ]

    # 支持的输出格式
    SUPPORTED_FORMATS = ["pdf", "musicxml", "mxl", "png", "svg", "mp3", "ogg"]

    def __init__(self, musescore_path: Optional[str] = None):
        """
        初始化 MuseScore 转换器

        Args:
            musescore_path: MuseScore 可执行文件路径 (可选)
        """
        self.musescore_path = musescore_path or self._find_musescore()

        if not self.musescore_path:
            logger.warning(
                "MuseScore not found. PDF generation will be unavailable."
            )

    def _find_musescore(self) -> Optional[str]:
        """
        查找 MuseScore 可执行文件

        Returns:
            Optional[str]: MuseScore 路径，未找到返回 None
        """
        for path in self.MUSESCORE_PATHS:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Found MuseScore at: {path}")
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def is_available(self) -> bool:
        """
        检查 MuseScore 是否可用

        Returns:
            bool: MuseScore 是否可用
        """
        return self.musescore_path is not None

    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: Optional[str] = None,
        timeout: int = 120
    ) -> bool:
        """
        转换文件格式

        Args:
            input_path: 输入文件路径 (MIDI, MusicXML 等)
            output_path: 输出文件路径
            output_format: 输出格式 (从 output_path 推断，如不指定)
            timeout: 超时时间 (秒)

        Returns:
            bool: 是否转换成功
        """
        if not self.is_available():
            logger.error("MuseScore is not available")
            return False

        # 验证输入文件
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return False

        # 推断输出格式
        if output_format is None:
            output_format = Path(output_path).suffix.lstrip(".")

        if output_format.lower() not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported output format: {output_format}")
            return False

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {input_path} to {output_path}")

        try:
            # 构建命令
            # 使用 xvfb-run 在无头环境中运行
            cmd = self._build_command(input_path, output_path)

            # 执行转换
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._get_env()
            )

            if result.returncode == 0:
                if os.path.exists(output_path):
                    logger.info(f"Conversion successful: {output_path}")
                    return True
                else:
                    logger.error("Conversion completed but output file not found")
                    return False
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Conversion timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.exception(f"Conversion error: {e}")
            return False

    def _build_command(self, input_path: str, output_path: str) -> List[str]:
        """
        构建 MuseScore 命令

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            List[str]: 命令参数列表
        """
        # 检查是否需要使用 xvfb-run (无显示器环境)
        use_xvfb = os.environ.get("DISPLAY") is None

        cmd = []

        if use_xvfb:
            # 使用 xvfb-run 创建虚拟显示器
            cmd.extend([
                "xvfb-run",
                "-a",  # 自动选择可用的显示器号
                "--server-args=-screen 0 1024x768x24"
            ])

        cmd.extend([
            self.musescore_path,
            "-o", output_path,  # 输出文件
            input_path  # 输入文件
        ])

        return cmd

    def _get_env(self) -> dict:
        """
        获取运行环境变量

        Returns:
            dict: 环境变量字典
        """
        env = os.environ.copy()

        # Qt 无头模式设置
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")

        return env

    def midi_to_pdf(
        self,
        midi_path: str,
        pdf_path: str,
        timeout: int = 120
    ) -> bool:
        """
        将 MIDI 文件转换为 PDF 乐谱

        Args:
            midi_path: MIDI 文件路径
            pdf_path: PDF 输出路径
            timeout: 超时时间

        Returns:
            bool: 是否成功
        """
        return self.convert(midi_path, pdf_path, "pdf", timeout)

    def midi_to_musicxml(
        self,
        midi_path: str,
        xml_path: str,
        timeout: int = 120
    ) -> bool:
        """
        将 MIDI 文件转换为 MusicXML

        Args:
            midi_path: MIDI 文件路径
            xml_path: MusicXML 输出路径
            timeout: 超时时间

        Returns:
            bool: 是否成功
        """
        return self.convert(midi_path, xml_path, "musicxml", timeout)


# 全局转换器实例
_converter: Optional[MuseScoreConverter] = None


def get_converter() -> MuseScoreConverter:
    """
    获取 MuseScore 转换器单例

    Returns:
        MuseScoreConverter: 转换器实例
    """
    global _converter

    if _converter is None:
        _converter = MuseScoreConverter()

    return _converter


def midi_to_pdf(midi_path: str, pdf_path: str, timeout: int = 120) -> bool:
    """
    便捷函数: MIDI 转 PDF

    Args:
        midi_path: MIDI 文件路径
        pdf_path: PDF 输出路径
        timeout: 超时时间

    Returns:
        bool: 是否成功
    """
    return get_converter().midi_to_pdf(midi_path, pdf_path, timeout)
