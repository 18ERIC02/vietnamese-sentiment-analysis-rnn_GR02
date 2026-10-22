"""Custom Qt widgets for the Vietnamese RNN Sentiment Analysis Desktop Application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    from ui.styles import (
        COLOR_AWAITING,
        COLOR_AWAITING_BG,
        COLOR_AWAITING_BORDER,
        COLOR_BG,
        COLOR_BORDER,
        COLOR_CARD,
        COLOR_MUTED,
        COLOR_NAVY,
        COLOR_NEGATIVE,
        COLOR_NEGATIVE_BG,
        COLOR_NEGATIVE_BORDER,
        COLOR_NEUTRAL,
        COLOR_NEUTRAL_BG,
        COLOR_NEUTRAL_BORDER,
        COLOR_POSITIVE,
        COLOR_POSITIVE_BG,
        COLOR_POSITIVE_BORDER,
        COLOR_PRIMARY,
        COLOR_TEXT,
    )
except ImportError:
    from desktop_app.ui.styles import (
        COLOR_AWAITING,
        COLOR_AWAITING_BG,
        COLOR_AWAITING_BORDER,
        COLOR_BG,
        COLOR_BORDER,
        COLOR_CARD,
        COLOR_MUTED,
        COLOR_NAVY,
        COLOR_NEGATIVE,
        COLOR_NEGATIVE_BG,
        COLOR_NEGATIVE_BORDER,
        COLOR_NEUTRAL,
        COLOR_NEUTRAL_BG,
        COLOR_NEUTRAL_BORDER,
        COLOR_POSITIVE,
        COLOR_POSITIVE_BG,
        COLOR_POSITIVE_BORDER,
        COLOR_PRIMARY,
        COLOR_TEXT,
    )


def QPointF_safe(x: float, y: float) -> QPointF:
    return QPointF(x, y)


class SectionIconBox(QWidget):
    """Small rounded square container with an icon matching the academic theme."""

    def __init__(self, icon_type: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.icon_type = icon_type
        self.setFixedSize(30, 30)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Soft blue container background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#E8F1FA")))
        painter.drawRoundedRect(0, 0, 30, 30, 8, 8)

        # Blue icon
        pen = QPen(QColor(COLOR_NAVY), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self.icon_type == "academic":
            # Graduation cap
            p = QPainterPath()
            p.moveTo(15, 8)
            p.lineTo(24, 12.5)
            p.lineTo(15, 17)
            p.lineTo(6, 12.5)
            p.closeSubpath()
            painter.drawPath(p)
            painter.drawLine(9, 14.5, 9, 21)
            p2 = QPainterPath()
            p2.moveTo(9, 18)
            p2.quadTo(15, 22, 21, 18)
            painter.drawPath(p2)
        elif self.icon_type == "book":
            # Open book
            painter.drawRect(7, 8, 16, 14)
            painter.drawLine(12, 8, 12, 22)
            painter.drawLine(15, 12, 20, 12)
            painter.drawLine(15, 16, 20, 16)
        elif self.icon_type == "user":
            # Teacher / person
            painter.drawEllipse(11, 7, 8, 8)
            p = QPainterPath()
            p.moveTo(7, 23)
            p.quadTo(15, 16, 23, 23)
            painter.drawPath(p)
        elif self.icon_type == "users":
            # Group of 3 members
            painter.drawEllipse(11, 7, 8, 8)
            p1 = QPainterPath()
            p1.moveTo(7, 23)
            p1.quadTo(15, 16, 23, 23)
            painter.drawPath(p1)
            # Left small person
            painter.drawEllipse(6, 9, 4.5, 4.5)
            # Right small person
            painter.drawEllipse(19.5, 9, 4.5, 4.5)
        elif self.icon_type == "doc":
            # Document
            p = QPainterPath()
            p.moveTo(9, 7)
            p.lineTo(17.5, 7)
            p.lineTo(21, 10.5)
            p.lineTo(21, 23)
            p.lineTo(9, 23)
            p.closeSubpath()
            painter.drawPath(p)
            painter.drawLine(12, 13, 18, 13)
            painter.drawLine(12, 17, 18, 17)
        elif self.icon_type == "gear":
            # Gear / Settings
            painter.drawEllipse(11, 11, 8, 8)
            painter.drawLine(15, 6.5, 15, 9)
            painter.drawLine(15, 21, 15, 23.5)
            painter.drawLine(6.5, 15, 9, 15)
            painter.drawLine(21, 15, 23.5, 15)
            painter.drawLine(9, 9, 11, 11)
            painter.drawLine(19, 19, 21, 21)
            painter.drawLine(9, 21, 11, 19)
            painter.drawLine(19, 11, 21, 9)
        elif self.icon_type == "chat":
            # Chat bubble
            p = QPainterPath()
            p.moveTo(7, 8)
            p.lineTo(23, 8)
            p.lineTo(23, 18)
            p.lineTo(13.5, 18)
            p.lineTo(10, 21.5)
            p.lineTo(10, 18)
            p.lineTo(7, 18)
            p.closeSubpath()
            painter.drawPath(p)
            painter.drawLine(11, 13, 19, 13)
        elif self.icon_type == "chart":
            # Bar chart
            painter.drawRect(6, 6, 18, 18)
            painter.drawLine(10, 20, 10, 15)
            painter.drawLine(15, 20, 15, 10)
            painter.drawLine(20, 20, 20, 12)
        elif self.icon_type == "info":
            # Circle with 'i'
            painter.drawEllipse(8, 8, 14, 14)
            painter.drawLine(15, 13.5, 15, 18)
            painter.drawPoint(15, 11.5)


class IconCircleWidget(QWidget):
    """Draws a crisp circular emotion icon with vector expressions."""

    def __init__(self, sentiment_type: str = "awaiting", size: int = 56, parent: QWidget | None = None):
        super().__init__(parent)
        self.sentiment_type = sentiment_type  # "positive", "neutral", "negative", "awaiting"
        self._size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def set_sentiment(self, sentiment_type: str) -> None:
        self.sentiment_type = sentiment_type
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        s = float(self._size)
        rect = QRectF(2, 2, s - 4, s - 4)

        if self.sentiment_type == "positive":
            bg_color = QColor(COLOR_POSITIVE)
            face_color = QColor("#FFFFFF")
        elif self.sentiment_type == "neutral":
            bg_color = QColor(COLOR_NEUTRAL)
            face_color = QColor("#FFFFFF")
        elif self.sentiment_type == "negative":
            bg_color = QColor(COLOR_NEGATIVE)
            face_color = QColor("#FFFFFF")
        else:
            bg_color = QColor(COLOR_MUTED)
            face_color = QColor("#FFFFFF")

        # Background circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawEllipse(rect)

        # Draw eyes and mouth
        cx = s / 2.0
        cy = s / 2.0
        r = (s - 4) / 2.0

        painter.setPen(QPen(face_color, max(2.2, s * 0.07), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        eye_offset_x = r * 0.38
        eye_y = cy - r * 0.16
        eye_radius = max(2.0, s * 0.055)

        # Eyes
        painter.setBrush(QBrush(face_color))
        painter.drawEllipse(QPointF_safe(cx - eye_offset_x, eye_y), eye_radius, eye_radius)
        painter.drawEllipse(QPointF_safe(cx + eye_offset_x, eye_y), eye_radius, eye_radius)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Mouth
        mouth_y = cy + r * 0.28
        mouth_half_w = r * 0.42

        if self.sentiment_type == "positive":
            # Upward smiling arc
            path = QPainterPath()
            path.moveTo(cx - mouth_half_w, mouth_y - r * 0.08)
            path.quadTo(cx, mouth_y + r * 0.32, cx + mouth_half_w, mouth_y - r * 0.08)
            painter.drawPath(path)
        elif self.sentiment_type == "negative":
            # Downward frowning arc
            path = QPainterPath()
            path.moveTo(cx - mouth_half_w, mouth_y + r * 0.18)
            path.quadTo(cx, mouth_y - r * 0.18, cx + mouth_half_w, mouth_y + r * 0.18)
            painter.drawPath(path)
        elif self.sentiment_type == "neutral":
            # Straight line
            painter.drawLine(int(cx - mouth_half_w), int(mouth_y), int(cx + mouth_half_w), int(mouth_y))
        else:
            # Subtle dash
            painter.drawLine(int(cx - r * 0.2), int(mouth_y), int(cx + r * 0.2), int(mouth_y))


class SentimentProgressBar(QWidget):
    """Custom smooth probability bar with label, track, and percentage text aligned to same axis."""

    def __init__(self, label: str, color_hex: str, icon_type: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.color_hex = color_hex
        self.label_text = label
        self._value = 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        # Icon
        self.icon_widget = IconCircleWidget(sentiment_type=icon_type, size=22, parent=self)
        layout.addWidget(self.icon_widget)

        # Fixed-width label for perfect alignment
        self.name_label = QLabel(label, self)
        self.name_label.setFixedWidth(72)
        self.name_label.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {COLOR_TEXT}; background: transparent;")
        layout.addWidget(self.name_label)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #E9EEF4;
                border: none;
                border-radius: 5px;
            }}
            QProgressBar::chunk {{
                background-color: {color_hex};
                border-radius: 5px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        # Fixed-width value label aligned to right
        self.value_label = QLabel("0.0%", self)
        self.value_label.setFixedWidth(52)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {COLOR_NAVY}; background: transparent;")
        layout.addWidget(self.value_label)

    def set_probability(self, prob: float) -> None:
        """Update probability value (0.0 to 1.0)."""
        self._value = max(0.0, min(1.0, float(prob)))
        int_val = int(self._value * 1000)
        self.progress_bar.setValue(int_val)
        self.value_label.setText(f"{self._value * 100.0:.1f}%")


class HeroSentimentBox(QFrame):
    """The prominent hero card on the left side of Result card."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("heroBox")
        self.setFixedHeight(136)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(14)

        # Big circular icon
        self.icon_widget = IconCircleWidget(sentiment_type="awaiting", size=62, parent=self)
        main_layout.addWidget(self.icon_widget, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Text column
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.caption_label = QLabel("Cảm xúc dự đoán", self)
        self.caption_label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {COLOR_MUTED}; background: transparent; border: none;")
        text_layout.addWidget(self.caption_label)

        self.sentiment_label = QLabel("Chờ phân tích", self)
        self.sentiment_label.setStyleSheet(f"font-size: 21px; font-weight: 800; color: {COLOR_MUTED}; background: transparent; border: none;")
        text_layout.addWidget(self.sentiment_label)

        # Changed from 'Độ tin cậy' to 'Xác suất dự đoán' per instructions
        self.confidence_label = QLabel("Xác suất dự đoán: —", self)
        self.confidence_label.setStyleSheet(f"font-size: 12.5px; font-weight: 600; color: {COLOR_TEXT}; background: transparent; border: none;")
        text_layout.addWidget(self.confidence_label)

        main_layout.addLayout(text_layout)
        self.set_state("awaiting", "Chờ phân tích", 0.0)

    def set_state(self, sentiment_type: str, label_text: str, confidence: float) -> None:
        """Update hero card styling based on predicted sentiment."""
        self.icon_widget.set_sentiment(sentiment_type)

        if sentiment_type == "positive":
            bg = COLOR_POSITIVE_BG
            border = COLOR_POSITIVE_BORDER
            text_color = COLOR_POSITIVE
        elif sentiment_type == "neutral":
            bg = COLOR_NEUTRAL_BG
            border = COLOR_NEUTRAL_BORDER
            text_color = COLOR_NEUTRAL
        elif sentiment_type == "negative":
            bg = COLOR_NEGATIVE_BG
            border = COLOR_NEGATIVE_BORDER
            text_color = COLOR_NEGATIVE
        else:
            bg = COLOR_AWAITING_BG
            border = COLOR_AWAITING_BORDER
            text_color = COLOR_AWAITING

        self.setStyleSheet(f"""
            QFrame#heroBox {{
                background-color: {bg};
                border: 1.5px solid {border};
                border-radius: 12px;
            }}
            QFrame#heroBox QLabel {{
                background: transparent;
                border: none;
            }}
        """)

        self.sentiment_label.setText(label_text)
        self.sentiment_label.setStyleSheet(f"font-size: 21px; font-weight: 800; color: {text_color}; background: transparent; border: none;")

        if confidence > 0.0:
            self.confidence_label.setText(f"Xác suất dự đoán: {confidence * 100.0:.1f}%")
        else:
            self.confidence_label.setText("Xác suất dự đoán: —")


class Badge(QLabel):
    """Rounded tech stack badge."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setProperty("class", "tech-badge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            QLabel.tech-badge {{
                background-color: #F1F6FB;
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 9px;
                padding: 3px 9px;
                font-size: 11px;
                font-weight: 600;
            }}
        """)


class InfoRowWidget(QWidget):
    """Sidebar info row with icon and title/value."""

    def __init__(self, title: str, value: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 2)
        layout.setSpacing(1)

        lbl_title = QLabel(title, self)
        lbl_title.setStyleSheet(f"font-size: 11px; color: {COLOR_MUTED}; font-weight: 600; background: transparent;")
        layout.addWidget(lbl_title)

        lbl_val = QLabel(value, self)
        lbl_val.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT}; font-weight: 600; background: transparent;")
        lbl_val.setWordWrap(True)
        layout.addWidget(lbl_val)


class HeaderWidget(QFrame):
    """University Header banner with seamless background transition and zero visible cut."""

    def __init__(self, logo_path: Path | None, banner_path: Path | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("headerFrame")
        self.setFixedHeight(120)
        self.logo_path = logo_path
        self.banner_path = banner_path
        self._banner_pixmap: Optional[QPixmap] = None

        if self.banner_path and self.banner_path.is_file():
            self._banner_pixmap = QPixmap(str(self.banner_path))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 0, 28, 0)
        layout.setSpacing(16)

        # HCM-UTE Logo
        self.logo_label = QLabel(self)
        self.logo_label.setFixedSize(68, 68)
        self.logo_label.setScaledContents(True)
        self.logo_label.setStyleSheet("background: transparent;")
        if logo_path and logo_path.is_file():
            self.logo_label.setPixmap(QPixmap(str(logo_path)))
        else:
            self.logo_label.setText("HCM-UTE")
            self.logo_label.setStyleSheet(f"font-weight: bold; color: {COLOR_NAVY}; background: transparent;")
        layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        # University Title text
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        lbl1 = QLabel("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT", self)
        lbl1.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {COLOR_NAVY}; letter-spacing: 0.5px; background: transparent;")
        text_layout.addWidget(lbl1)

        lbl2 = QLabel("THÀNH PHỐ HỒ CHÍ MINH", self)
        lbl2.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {COLOR_NAVY}; letter-spacing: 0.5px; background: transparent;")
        text_layout.addWidget(lbl2)

        lbl3 = QLabel("Ho Chi Minh City University of Technology and Engineering", self)
        lbl3.setStyleSheet(f"font-size: 12px; color: {COLOR_MUTED}; font-weight: 500; background: transparent;")
        text_layout.addWidget(lbl3)

        layout.addLayout(text_layout)
        layout.addStretch()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        # 1. Base smooth background gradient from left to right
        bg_grad = QLinearGradient(0, 0, w, 0)
        bg_grad.setColorAt(0.0, QColor("#FFFFFF"))
        bg_grad.setColorAt(0.38, QColor("#FFFFFF"))
        bg_grad.setColorAt(0.70, QColor("#F4F8FD"))
        bg_grad.setColorAt(1.0, QColor("#EAF3FB"))
        painter.fillRect(self.rect(), bg_grad)

        # 2. Draw building banner on the right side with alpha feathering
        if self._banner_pixmap and not self._banner_pixmap.isNull():
            orig_w = self._banner_pixmap.width()
            orig_h = self._banner_pixmap.height()
            scaled_w = int(orig_w * (h / orig_h))
            target_x = w - scaled_w
            painter.drawPixmap(target_x, 0, scaled_w, h, self._banner_pixmap)

        # 3. Clean subtle bottom border line
        painter.setPen(QPen(QColor(COLOR_BORDER), 1.0))
        painter.drawLine(0, h - 1, w, h - 1)


class ModelLoadWorker(QThread):
    """Asynchronous worker for model loading so UI starts without freezing."""

    loaded = Signal(dict)
    failed = Signal(str)

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def run(self):
        try:
            self.predictor.load()
            self.loaded.emit(self.predictor.metadata)
        except Exception as e:
            self.failed.emit(str(e))


class InferenceWorker(QThread):
    """Asynchronous worker for running model inference without UI freezing."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, predictor, text: str):
        super().__init__()
        self.predictor = predictor
        self.text = text

    def run(self):
        try:
            result = self.predictor.predict(self.text)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
