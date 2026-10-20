"""Entry point for Vietnamese RNN Sentiment Analysis Desktop Application.

Project:
    XÂY DỰNG HỆ THỐNG PHÂN TÍCH CẢM XÚC BÌNH LUẬN TIẾNG VIỆT BẰNG RNN
    TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH (HCMUTE)
    Môn học: Deep Learning
    Giảng viên: PGS. TS. Hoàng Văn Dũng
    Nhóm thực hiện:
        1. Trịnh Nguyễn Anh Hào (2611307)
        2. Lê Huy Huân (2611308)
        3. Nguyễn Nam Triều Tiên (2611323)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Setup import paths reliably for both source execution and frozen PyInstaller bundles
APP_DIR = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    base_dir = Path(getattr(sys, "_MEIPASS", APP_DIR))
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    exe_dir = Path(sys.executable).resolve().parent
    if str(exe_dir) not in sys.path:
        sys.path.insert(0, str(exe_dir))
    internal_dir = exe_dir / "_internal"
    if internal_dir.is_dir() and str(internal_dir) not in sys.path:
        sys.path.insert(0, str(internal_dir))
else:
    PROJECT_ROOT = APP_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

# Suppress excessive TensorFlow logs on desktop startup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

try:
    from ui.main_window import MainWindow
except ImportError:
    from desktop_app.ui.main_window import MainWindow


def main() -> int:
    """Launch the PySide6 Desktop application."""
    # Enable crisp text rendering on High-DPI Windows monitors
    if hasattr(Qt.HighDpiScaleFactorRoundingPolicy, "PassThrough"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)
    app.setApplicationName("Phân tích cảm xúc RNN | HCMUTE")
    app.setOrganizationName("HCMUTE")
    app.setOrganizationDomain("hcmute.edu.vn")

    logo_path = APP_DIR / "assets" / "hcmute_logo.png"
    if logo_path.is_file():
        app.setWindowIcon(QIcon(str(logo_path)))

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
