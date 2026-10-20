"""Main window of the Vietnamese RNN Sentiment Analysis Desktop Application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from inference.predictor import SentimentPredictor
    from ui.styles import (
        APP_QSS,
        COLOR_BG,
        COLOR_BORDER,
        COLOR_CARD,
        COLOR_MUTED,
        COLOR_NAVY,
        COLOR_PRIMARY,
        COLOR_TEXT,
    )
    from ui.widgets import (
        Badge,
        HeaderWidget,
        HeroSentimentBox,
        InferenceWorker,
        InfoRowWidget,
        ModelLoadWorker,
        SectionIconBox,
        SentimentProgressBar,
    )
except ImportError:
    from desktop_app.inference.predictor import SentimentPredictor
    from desktop_app.ui.styles import (
        APP_QSS,
        COLOR_BG,
        COLOR_BORDER,
        COLOR_CARD,
        COLOR_MUTED,
        COLOR_NAVY,
        COLOR_PRIMARY,
        COLOR_TEXT,
    )
    from desktop_app.ui.widgets import (
        Badge,
        HeaderWidget,
        HeroSentimentBox,
        InferenceWorker,
        InfoRowWidget,
        ModelLoadWorker,
        SectionIconBox,
        SentimentProgressBar,
    )


def get_asset_path(filename: str) -> Path:
    """Find asset path across dev source and PyInstaller frozen directories."""
    app_dir = Path(__file__).resolve().parent.parent
    candidates = [
        app_dir / "assets" / filename,
    ]
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            candidates.insert(0, Path(sys._MEIPASS) / "assets" / filename)
        exe_dir = Path(sys.executable).resolve().parent
        candidates.insert(0, exe_dir / "assets" / filename)
        candidates.insert(0, exe_dir / "_internal" / "assets" / filename)

    for c in candidates:
        if c.is_file():
            return c.resolve()
    return candidates[0]


LOGO_PATH = get_asset_path("hcmute_logo.png")
BANNER_PATH = get_asset_path("banner_feathered.png")
if not BANNER_PATH.is_file():
    BANNER_PATH = get_asset_path("banner.png")
CHAT_DECOR_PATH = get_asset_path("chat_decor.png")

MAX_CHARS = 500

EXAMPLES = [
    "Thầy giảng rất dễ hiểu.",
    "Bài giảng quá khó hiểu.",
    "Buổi học bắt đầu lúc 7 giờ.",
]


class MainWindow(QMainWindow):
    """Main window class for the Sentiment Analysis Desktop App."""

    def __init__(self, predictor: SentimentPredictor | None = None):
        super().__init__()
        self.predictor = predictor or SentimentPredictor()
        self.current_inference_worker: Optional[InferenceWorker] = None
        self.load_worker: Optional[ModelLoadWorker] = None

        self.setWindowTitle("Phân tích cảm xúc bình luận tiếng Việt | HCMUTE")
        self.resize(1360, 820)
        self.setMinimumSize(1080, 680)

        if LOGO_PATH.is_file():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self.setStyleSheet(APP_QSS)
        self._init_ui()
        self._start_model_loading()

    def _init_ui(self) -> None:
        """Construct the entire desktop interface with clean spacing and responsive layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Header Frame spanning full width (Custom painted, seamless banner)
        self.header_widget = HeaderWidget(LOGO_PATH, BANNER_PATH, self)
        root_layout.addWidget(self.header_widget)

        # 2. Scrollable Body Area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        body_container = QWidget()
        body_container.setStyleSheet(f"background-color: {COLOR_BG};")
        body_layout = QVBoxLayout(body_container)
        body_layout.setContentsMargins(20, 14, 20, 16)
        body_layout.setSpacing(0)
        body_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 2b. Columns Container (Side-by-side equal-height layout)
        columns_container = QWidget(body_container)
        columns_container.setStyleSheet("background: transparent;")
        columns_layout = QHBoxLayout(columns_container)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(18)

        # 3. Sidebar (Left column, width ~310px, expanding vertically to match main content)
        self.sidebar_widget = self._create_sidebar()
        self.sidebar_widget.setFixedWidth(310)
        self.sidebar_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        columns_layout.addWidget(self.sidebar_widget)

        # 4. Main Content (Right column, expanding horizontally)
        self.content_widget = self._create_main_content()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        columns_layout.addWidget(self.content_widget)

        body_layout.addWidget(columns_container)
        body_layout.addStretch()

        scroll_area.setWidget(body_container)
        root_layout.addWidget(scroll_area)

    def _create_sidebar(self) -> QWidget:
        """Create the left sidebar with Project Info, Intro, and Tech stack."""
        sidebar = QFrame(self)
        sidebar.setProperty("class", "dashboard-card")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 14, 16, 14)
        sidebar_layout.setSpacing(10)

        # --- Section 1: Thông tin đồ án ---
        head1 = QHBoxLayout()
        head1.setSpacing(8)
        head1.addWidget(SectionIconBox("academic", sidebar))
        lbl_info = QLabel("Thông tin đồ án", sidebar)
        lbl_info.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {COLOR_NAVY}; background: transparent;")
        head1.addWidget(lbl_info)
        head1.addStretch()
        sidebar_layout.addLayout(head1)

        # Sub-item: Môn học
        row_sub = QHBoxLayout()
        row_sub.setSpacing(8)
        row_sub.addWidget(SectionIconBox("book", sidebar))
        row_sub.addWidget(InfoRowWidget("Môn học", "Deep Learning", sidebar))
        sidebar_layout.addLayout(row_sub)

        # Sub-item: Giảng viên hướng dẫn
        row_gv = QHBoxLayout()
        row_gv.setSpacing(8)
        row_gv.addWidget(SectionIconBox("user", sidebar))
        row_gv.addWidget(InfoRowWidget("Giảng viên hướng dẫn", "PGS. TS. Hoàng Văn Dũng", sidebar))
        sidebar_layout.addLayout(row_gv)

        # Sub-item: Nhóm thực hiện
        row_team = QHBoxLayout()
        row_team.setSpacing(8)
        row_team.addWidget(SectionIconBox("users", sidebar))
        team_text = (
            "1. Trịnh Nguyễn Anh Hào (2611307)\n"
            "2. Lê Huy Huân (2611308)\n"
            "3. Nguyễn Nam Triều Tiên (2611323)"
        )
        row_team.addWidget(InfoRowWidget("Nhóm thực hiện", team_text, sidebar))
        sidebar_layout.addLayout(row_team)

        # Divider
        div1 = QFrame(sidebar)
        div1.setFrameShape(QFrame.Shape.HLine)
        div1.setStyleSheet(f"color: {COLOR_BORDER}; margin: 2px 0px;")
        sidebar_layout.addWidget(div1)

        # --- Section 2: Giới thiệu ---
        head2 = QHBoxLayout()
        head2.setSpacing(8)
        head2.addWidget(SectionIconBox("doc", sidebar))
        lbl_intro = QLabel("Giới thiệu", sidebar)
        lbl_intro.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {COLOR_NAVY}; background: transparent;")
        head2.addWidget(lbl_intro)
        head2.addStretch()
        sidebar_layout.addLayout(head2)

        lbl_desc = QLabel(
            "Ứng dụng sử dụng mạng nơ-ron hồi quy RNN để phân tích cảm xúc "
            "bình luận tiếng Việt, phân loại thành ba nhóm: Tiêu cực, "
            "Trung lập và Tích cực.",
            sidebar,
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet(f"font-size: 12.5px; color: {COLOR_MUTED}; line-height: 1.45; background: transparent;")
        sidebar_layout.addWidget(lbl_desc)

        # Divider
        div2 = QFrame(sidebar)
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet(f"color: {COLOR_BORDER}; margin: 2px 0px;")
        sidebar_layout.addWidget(div2)

        # --- Section 3: Công nghệ sử dụng ---
        head3 = QHBoxLayout()
        head3.setSpacing(8)
        head3.addWidget(SectionIconBox("gear", sidebar))
        lbl_tech = QLabel("Công nghệ sử dụng", sidebar)
        lbl_tech.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {COLOR_NAVY}; background: transparent;")
        head3.addWidget(lbl_tech)
        head3.addStretch()
        sidebar_layout.addLayout(head3)

        # Badges layout
        badges_row1 = QHBoxLayout()
        badges_row1.setSpacing(5)
        badges_row1.addWidget(Badge("Python", sidebar))
        badges_row1.addWidget(Badge("TensorFlow", sidebar))
        badges_row1.addWidget(Badge("RNN", sidebar))  # Strictly RNN!
        badges_row1.addStretch()
        sidebar_layout.addLayout(badges_row1)

        badges_row2 = QHBoxLayout()
        badges_row2.setSpacing(5)
        badges_row2.addWidget(Badge("PySide6", sidebar))
        badges_row2.addWidget(Badge("Vietnamese NLP", sidebar))
        badges_row2.addStretch()
        sidebar_layout.addLayout(badges_row2)

        sidebar_layout.addStretch()
        return sidebar

    def _create_main_content(self) -> QWidget:
        """Create the right main area with Title Banner, Input Card, and Result Card."""
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(11)

        # 1. Title Banner
        title_banner = self._create_title_banner()
        layout.addWidget(title_banner)

        # 2. Input Card (Compact height)
        input_card = self._create_input_card()
        layout.addWidget(input_card)

        # 3. Result Card (Balanced proportions)
        result_card = self._create_result_card()
        layout.addWidget(result_card)

        return content

    def _create_title_banner(self) -> QWidget:
        """Create the top title banner with accent underline and seamless transparent decor."""
        banner = QWidget(self)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(4, 0, 10, 0)
        layout.setSpacing(16)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        main_title = QLabel("Phân tích cảm xúc bình luận tiếng Việt", banner)
        main_title.setStyleSheet(f"font-size: 26px; font-weight: 800; color: {COLOR_NAVY}; letter-spacing: 0.2px; background: transparent;")
        text_layout.addWidget(main_title)

        sub_title = QLabel(
            "Ứng dụng mạng nơ-ron hồi quy RNN trong phân loại cảm xúc văn bản.", banner
        )
        sub_title.setStyleSheet(f"font-size: 13.5px; color: {COLOR_MUTED}; font-weight: 500; background: transparent;")
        text_layout.addWidget(sub_title)

        accent = QFrame(banner)
        accent.setFixedSize(48, 3)
        accent.setStyleSheet(f"background-color: {COLOR_PRIMARY}; border-radius: 1.5px;")
        text_layout.addWidget(accent)

        layout.addLayout(text_layout)
        layout.addStretch()

        # Chat graphic illustration with clean transparent background
        if CHAT_DECOR_PATH.is_file():
            decor_lbl = QLabel(banner)
            decor_lbl.setStyleSheet("background: transparent;")
            decor_pix = QPixmap(str(CHAT_DECOR_PATH))
            scaled_pix = decor_pix.scaledToHeight(74, Qt.TransformationMode.SmoothTransformation)
            decor_lbl.setPixmap(scaled_pix)
            layout.addWidget(decor_lbl, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return banner

    def _create_input_card(self) -> QWidget:
        """Create the input card for entering comments and triggering analysis (compact height)."""
        card = QFrame(self)
        card.setProperty("class", "dashboard-card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(9)

        # Card header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        header_layout.addWidget(SectionIconBox("chat", card))
        lbl_head = QLabel("Nhập bình luận", card)
        lbl_head.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {COLOR_NAVY}; background: transparent;")
        header_layout.addWidget(lbl_head)

        # Status badge in header
        self.status_label = QLabel("Đang tải mô hình...", card)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #FCD34D;
                border-radius: 11px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
        """)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        layout.addLayout(header_layout)

        # Text input with counter (compact 82px height)
        input_container = QVBoxLayout()
        input_container.setSpacing(2)

        self.comment_input = QTextEdit(card)
        self.comment_input.setObjectName("commentInput")
        self.comment_input.setPlaceholderText("Nhập bình luận tiếng Việt cần phân tích cảm xúc...")
        self.comment_input.setFixedHeight(82)
        self.comment_input.textChanged.connect(self._on_text_changed)
        input_container.addWidget(self.comment_input)

        counter_layout = QHBoxLayout()
        counter_layout.addStretch()
        self.char_counter = QLabel(f"0/{MAX_CHARS}", card)
        self.char_counter.setStyleSheet(f"font-size: 11px; color: {COLOR_MUTED}; background: transparent;")
        counter_layout.addWidget(self.char_counter)
        input_container.addLayout(counter_layout)
        layout.addLayout(input_container)

        # Example buttons row
        examples_layout = QHBoxLayout()
        examples_layout.setSpacing(8)

        lbl_try = QLabel("Thử với ví dụ:", card)
        lbl_try.setStyleSheet(f"font-size: 12.5px; font-weight: 600; color: {COLOR_TEXT}; background: transparent;")
        examples_layout.addWidget(lbl_try)

        for text in EXAMPLES:
            btn = QPushButton(text, card)
            btn.setProperty("class", "example-pill")
            btn.clicked.connect(lambda checked=False, t=text: self._set_example_text(t))
            examples_layout.addWidget(btn)

        examples_layout.addStretch()
        layout.addLayout(examples_layout)

        # Main Analyze Button (compact 40px height)
        self.analyze_button = QPushButton("▶   Phân tích cảm xúc", card)
        self.analyze_button.setFixedHeight(40)
        self.analyze_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 9px;
                font-size: 14px;
                font-weight: 700;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: #1565C0;
            }}
            QPushButton:pressed {{
                background-color: #0D47A1;
            }}
            QPushButton:disabled {{
                background-color: #CBD5E1;
                color: {COLOR_MUTED};
            }}
        """)
        self.analyze_button.setEnabled(False)  # Disabled until model loaded
        self.analyze_button.clicked.connect(self._run_analysis)
        layout.addWidget(self.analyze_button)

        return card

    def _create_result_card(self) -> QWidget:
        """Create the result card with hero sentiment (35%) and class probability bars (65%)."""
        card = QFrame(self)
        card.setProperty("class", "dashboard-card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        # Card header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        header_layout.addWidget(SectionIconBox("chart", card))
        lbl_head = QLabel("Kết quả phân tích", card)
        lbl_head.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {COLOR_NAVY}; background: transparent;")
        header_layout.addWidget(lbl_head)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Two-column result body with ~35% : 65% ratio
        body_split = QHBoxLayout()
        body_split.setSpacing(20)

        # Left: Hero sentiment box (35% stretch)
        self.hero_box = HeroSentimentBox(card)
        body_split.addWidget(self.hero_box, stretch=35, alignment=Qt.AlignmentFlag.AlignTop)

        # Right: Probability distribution bars (65% stretch)
        prob_container = QVBoxLayout()
        prob_container.setSpacing(6)

        lbl_dist = QLabel("Phân bố xác suất các lớp", card)
        lbl_dist.setStyleSheet(f"font-size: 13.5px; font-weight: 700; color: {COLOR_NAVY}; margin-bottom: 2px; background: transparent;")
        prob_container.addWidget(lbl_dist)

        self.bar_negative = SentimentProgressBar("Tiêu cực", "#EF4444", "negative", card)
        prob_container.addWidget(self.bar_negative)

        self.bar_neutral = SentimentProgressBar("Trung tính", "#3B82F6", "neutral", card)
        prob_container.addWidget(self.bar_neutral)

        self.bar_positive = SentimentProgressBar("Tích cực", "#10B981", "positive", card)
        prob_container.addWidget(self.bar_positive)

        body_split.addLayout(prob_container, stretch=65)
        layout.addLayout(body_split)

        # Remark Frame (Nhận xét)
        self.remark_frame = QFrame(card)
        self.remark_frame.setObjectName("remarkFrame")
        self.remark_frame.setStyleSheet("""
            QFrame#remarkFrame {
                background-color: #F0F6FD;
                border: 1px solid #CDE0F5;
                border-radius: 9px;
            }
            QFrame#remarkFrame QLabel {
                border: none;
                background: transparent;
            }
        """)
        remark_layout = QHBoxLayout(self.remark_frame)
        remark_layout.setContentsMargins(12, 8, 12, 8)
        remark_layout.setSpacing(10)

        remark_layout.addWidget(SectionIconBox("info", self.remark_frame), alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        lbl_rmk_title = QLabel("Nhận xét", self.remark_frame)
        lbl_rmk_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #0369A1;")
        text_col.addWidget(lbl_rmk_title)

        self.remark_text_label = QLabel(
            "Vui lòng nhập bình luận và nhấn 'Phân tích cảm xúc' để xem kết quả.",
            self.remark_frame,
        )
        self.remark_text_label.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT};")
        self.remark_text_label.setWordWrap(True)
        text_col.addWidget(self.remark_text_label)

        remark_layout.addLayout(text_col)
        layout.addWidget(self.remark_frame)

        return card

    # --- Event Handlers & Workers ---

    def _start_model_loading(self) -> None:
        """Trigger background model loading so window pops up instantly."""
        self.status_label.setText("Đang tải mô hình...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #FCD34D;
                border-radius: 11px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
        """)

        self.load_worker = ModelLoadWorker(self.predictor)
        self.load_worker.loaded.connect(self._on_model_loaded)
        self.load_worker.failed.connect(self._on_model_load_failed)
        self.load_worker.start()

    def _on_model_loaded(self, metadata: dict) -> None:
        """Handle successful model loading."""
        self.status_label.setText("Mô hình RNN sẵn sàng")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #DCFCE7;
                color: #166534;
                border: 1px solid #86EFAC;
                border-radius: 11px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
        """)
        self._update_button_state()

    def _on_model_load_failed(self, error_msg: str) -> None:
        """Handle model loading error."""
        self.status_label.setText("Lỗi nạp mô hình")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #FEE2E2;
                color: #991B1B;
                border: 1px solid #FCA5A5;
                border-radius: 11px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
        """)
        self.analyze_button.setEnabled(False)

        QMessageBox.critical(
            self,
            "Lỗi nạp mô hình",
            f"Không thể nạp mô hình RNN từ checkpoint:\n\n{error_msg}\n\n"
            "Vui lòng kiểm tra lại file checkpoint trong thư mục models.",
        )

    def _on_text_changed(self) -> None:
        """Update character count and validate button state."""
        text = self.comment_input.toPlainText()
        length = len(text)
        self.char_counter.setText(f"{length}/{MAX_CHARS}")

        if length > MAX_CHARS:
            self.char_counter.setStyleSheet("font-size: 11px; color: #EF4444; font-weight: bold; background: transparent;")
        else:
            self.char_counter.setStyleSheet(f"font-size: 11px; color: {COLOR_MUTED}; background: transparent;")

        self._update_button_state()

    def _update_button_state(self) -> None:
        """Enable or disable Analyze button depending on text and model status."""
        text = self.comment_input.toPlainText().strip()
        length = len(self.comment_input.toPlainText())
        is_ready = self.predictor.is_loaded and bool(text) and (length <= MAX_CHARS)
        self.analyze_button.setEnabled(is_ready)

    def _set_example_text(self, text: str) -> None:
        """Populate the text edit with an example and trigger button update."""
        self.comment_input.setPlainText(text)
        self.comment_input.moveCursor(self.comment_input.textCursor().MoveOperation.End)

    def _run_analysis(self) -> None:
        """Execute sentiment analysis on the current text."""
        text = self.comment_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(
                self,
                "Chưa nhập bình luận",
                "Vui lòng nhập nội dung bình luận tiếng Việt cần phân tích.",
            )
            return

        if len(self.comment_input.toPlainText()) > MAX_CHARS:
            QMessageBox.warning(
                self,
                "Vượt quá giới hạn ký tự",
                f"Bình luận có độ dài vượt quá giới hạn cho phép ({MAX_CHARS} ký tự).",
            )
            return

        # Disable button during inference
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("⏳   Đang phân tích...")

        # Run in worker thread
        self.current_inference_worker = InferenceWorker(self.predictor, text)
        self.current_inference_worker.finished.connect(self._on_analysis_finished)
        self.current_inference_worker.error.connect(self._on_analysis_error)
        self.current_inference_worker.start()

    def _on_analysis_finished(self, result: Dict[str, Any]) -> None:
        """Update result widgets with model outputs."""
        self.analyze_button.setText("▶   Phân tích cảm xúc")
        self._update_button_state()

        probs = result["probabilities"]  # [neg, neu, pos]
        class_idx = result["class_index"]
        confidence = result["confidence"]

        # Update Probability Bars
        self.bar_negative.set_probability(probs[0])
        self.bar_neutral.set_probability(probs[1])
        self.bar_positive.set_probability(probs[2])

        # Map index to state
        if class_idx == 2:
            sentiment_type = "positive"
            label_text = "Tích cực"
            remark = "Bình luận mang sắc thái tích cực theo dự đoán của mô hình."
        elif class_idx == 1:
            sentiment_type = "neutral"
            label_text = "Trung tính"
            remark = "Bình luận được dự đoán thuộc nhóm cảm xúc trung lập."
        else:
            sentiment_type = "negative"
            label_text = "Tiêu cực"
            remark = "Bình luận mang sắc thái tiêu cực theo dự đoán của mô hình."

        # Update Hero Box
        self.hero_box.set_state(sentiment_type, label_text, confidence)

        # Update Remark
        self.remark_text_label.setText(remark)

    def _on_analysis_error(self, error_msg: str) -> None:
        """Handle analysis error."""
        self.analyze_button.setText("▶   Phân tích cảm xúc")
        self._update_button_state()

        QMessageBox.warning(
            self,
            "Lỗi phân tích cảm xúc",
            f"Không thể phân tích bình luận:\n\n{error_msg}",
        )
