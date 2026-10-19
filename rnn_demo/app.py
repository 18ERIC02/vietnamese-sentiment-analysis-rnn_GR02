"""HCMUTE dashboard for inference with the trained Vietnamese SimpleRNN."""

from __future__ import annotations

import base64
import html
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import streamlit as st
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best_model.keras"
LOGO_PATH = APP_DIR / "assets" / "hcmute_logo.png"
CSS_PATH = APP_DIR / "assets" / "style.css"
LABELS_VI = ("Tiêu cực", "Trung lập", "Tích cực")
LABELS_EN = ("Negative", "Neutral", "Positive")
EXAMPLES = (
    "Thầy giảng rất dễ hiểu.",
    "Bài giảng quá khó hiểu.",
    "Buổi học bắt đầu lúc 7 giờ.",
)
TEAM = (
    ("2611307", "Trịnh Nguyễn Anh Hào"),
    ("2611308", "Lê Huy Huân"),
    ("2611323", "Nguyễn Nam Triều Tiên"),
)
ICONS = {
    "users": '<circle cx="9" cy="7" r="4"/><path d="M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2M16 3.1a4 4 0 0 1 0 7.8M22 21v-2a4 4 0 0 0-3-3.9"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M8 13h8M8 17h8"/>',
    "settings": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/>',
    "edit": '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L9 17l-4 1 1-4z"/>',
    "chart": '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M8 16v-4M12 16V8M16 16v-6"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 11v6M12 7h.01"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "code": '<path d="m8 4-6 8 6 8M16 4l6 8-6 8"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5zM3 12l9 5 9-5M3 17l9 5 9-5"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "smile": '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/>',
    "meh": '<circle cx="12" cy="12" r="10"/><path d="M8 15h8M9 9h.01M15 9h.01"/>',
    "frown": '<circle cx="12" cy="12" r="10"/><path d="M8 16s1.5-2 4-2 4 2 4 2M9 9h.01M15 9h.01"/>',
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def icon(name: str, size: int = 20) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


def heading(name: str, title: str, subtitle: str = "", tone: str = "blue") -> str:
    note = f'<p>{esc(subtitle)}</p>' if subtitle else ""
    return (f'<div class="card-heading"><span class="icon-box {tone}">{icon(name)}</span>'
            f'<div><h2>{esc(title)}</h2>{note}</div></div>')


def row(label: str, value: object) -> str:
    return f'<div class="info-row"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def bar(index: int, value: float, compact: bool = False) -> str:
    width = max(0.0, min(100.0, float(value) * 100.0))
    variant = " compact" if compact else ""
    return (f'<div class="prob-row{variant}"><span>{LABELS_EN[index]}</span>'
            f'<div class="prob-track" role="progressbar" aria-label="{LABELS_VI[index]}" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{width:.4f}">'
            f'<div class="prob-fill sentiment-{index}" style="width:{width:.4f}%"></div></div>'
            f'<strong>{pct(value)}</strong></div>')


@st.cache_resource(show_spinner="Đang nạp mô hình RNN...")
def load_sentiment_model():
    # Keras 3.13 opens Vietnamese vocabulary assets without an encoding on Windows.
    original_open = open

    def open_vocabulary_as_utf8(file, mode="r", *args, **kwargs):
        is_vocabulary = (
            isinstance(file, (str, bytes, Path))
            and Path(file).name == "vocabulary.txt"
        )
        if is_vocabulary and "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original_open(file, mode, *args, **kwargs)

    with patch("builtins.open", open_vocabulary_as_utf8):
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def get_metadata(model) -> dict:
    by_type = {layer.__class__.__name__: layer for layer in model.layers}
    vectorizer = by_type["TextVectorization"]
    return {
        "vectorizer": vectorizer,
        "vocabulary_size": int(vectorizer.vocabulary_size()),
        "sequence_length": int(vectorizer.get_config()["output_sequence_length"]),
        "embedding_dim": int(by_type["Embedding"].output_dim),
        "rnn_units": int(by_type["SimpleRNN"].units),
        "dropout": float(by_type["Dropout"].rate),
        "classes": int(model.output_shape[-1]),
    }


def inspect_pipeline(vectorizer, input_tensor) -> dict:
    """Use the checkpoint's own preprocessing and lookup; never invent token IDs."""
    data = {"normalized": None, "tokens": None, "raw_ids": None, "padded_ids": None}
    try:
        preprocessed = vectorizer._preprocess(input_tensor)
        tokens = [part.decode("utf-8") for part in preprocessed.to_list()[0]]
        data["tokens"] = tokens
        data["normalized"] = " ".join(tokens)
        data["raw_ids"] = [
            int(value) for value in vectorizer._lookup_layer.call(preprocessed).to_list()[0]
        ]
    except (AttributeError, TypeError, ValueError):
        pass  # Private inspection APIs vary by Keras version.
    try:
        data["padded_ids"] = (
            np.asarray(vectorizer(input_tensor)).reshape(-1).astype(int).tolist()
        )
    except (TypeError, ValueError):
        pass
    return data


def run_inference(model, metadata: dict, text: str) -> dict:
    input_tensor = tf.constant([[text]], dtype=tf.string)
    started = time.perf_counter()
    output = model(input_tensor, training=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities = np.asarray(output, dtype=np.float64)
    if probabilities.shape != (1, 3):
        raise ValueError(f"Đầu ra cần shape (1, 3), nhận {probabilities.shape}.")
    probabilities = probabilities[0]
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Xác suất mô hình không hợp lệ.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError("Tổng ba xác suất không xấp xỉ 1.")
    return {
        "text": text,
        "probabilities": probabilities.tolist(),
        "class_index": int(np.argmax(probabilities)),
        "elapsed_ms": elapsed_ms,
        "pipeline": inspect_pipeline(metadata["vectorizer"], input_tensor),
    }


def display_ids(values: list[int] | None) -> str:
    if values is None:
        return "Không khả dụng trong phiên bản Keras hiện tại."
    if len(values) <= 30:
        return "[" + ", ".join(map(str, values)) + "]"
    start = ", ".join(map(str, values[:22]))
    end = ", ".join(map(str, values[-8:]))
    return f"[{start}, … {len(values) - 30} vị trí ẩn …, {end}] · tổng {len(values)} ID"


def timeline_step(number: int, title: str, note: str, value_html: str) -> str:
    final = " final" if number == 8 else ""
    return (f'<div class="timeline-row"><span class="timeline-badge{final}">{number}</span>'
            f'<div class="timeline-copy"><h3>{esc(title)}</h3><p>{esc(note)}</p></div>'
            f'<div class="timeline-value">{value_html}</div></div>')


def render_results(result: dict, metadata: dict) -> None:
    probabilities = result["probabilities"]
    index = result["class_index"]
    confidence = probabilities[index]
    pipeline = result["pipeline"]
    token_count = len(pipeline["tokens"]) if pipeline["tokens"] is not None else "Không khả dụng"
    prediction = (
        '<div class="dashboard-card result-card">'
        + heading("chart", "Kết quả dự đoán", "Kết quả phân tích cảm xúc của mô hình RNN cho văn bản đầu vào.")
        + f'<div class="prediction-highlight sentiment-bg-{index}">'
        + f'<span class="prediction-icon">{icon(("frown", "meh", "smile")[index], 31)}</span>'
        + f'<div class="prediction-name"><small>Cảm xúc dự đoán</small><strong>{LABELS_EN[index]}</strong>'
        + f'<span>{LABELS_VI[index]}</span></div>'
        + f'<div class="prediction-confidence"><small>Xác suất cao nhất</small>'
        + f'<strong>{pct(confidence)}</strong></div></div>'
        + "".join(bar(i, p) for i, p in enumerate(probabilities))
        + '<p class="fine-note">Xác suất Softmax chưa được hiệu chuẩn; không diễn giải như độ tin cậy đã kiểm chứng.</p>'
        + '</div>'
    )
    details = (
        row("Văn bản đầu vào", result["text"])
        + row("Số token", token_count)
        + row("Độ dài chuỗi sau padding", metadata["sequence_length"])
        + row("Lớp dự đoán", f'{LABELS_EN[index]} · {LABELS_VI[index]}')
        + row("Xác suất cao nhất", pct(confidence))
        + row("Thời gian inference", f'{result["elapsed_ms"]:.2f} ms')
    )
    inference = ('<div class="dashboard-card result-card">'
                 + heading("info", "Thông tin inference")
                 + f'<div class="inference-table">{details}</div></div>')
    st.markdown(f'<div class="result-grid">{prediction}{inference}</div>', unsafe_allow_html=True)


def render_stats(metadata: dict, result: dict | None) -> None:
    stats = (
        ("database", "Kích thước từ vựng", f'{metadata["vocabulary_size"]:,}', "Token trong vocabulary đã lưu", "mint"),
        ("code", "Độ dài chuỗi tối đa", str(metadata["sequence_length"]), "Vị trí sau vector hóa", "blue"),
        ("layers", "Số lớp đầu ra", str(metadata["classes"]), "Tiêu cực / Trung lập / Tích cực", "mint"),
        ("clock", "Thời gian inference", f'{result["elapsed_ms"]:.2f} ms' if result else "—", "Lần dự đoán gần nhất", "mint"),
    )
    cards = "".join(
        '<div class="dashboard-card stat-card">'
        + f'<span class="icon-box {tone}">{icon(icon_name)}</span>'
        + f'<div><small>{esc(title)}</small><strong>{esc(value)}</strong><span>{esc(note)}</span></div>'
        + '</div>'
        for icon_name, title, value, note, tone in stats
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)


def render_timeline(result: dict) -> None:
    data = result["pipeline"]
    tokens = data["tokens"]
    index = result["class_index"]
    probabilities = result["probabilities"]
    token_html = (
        '<div class="token-list">' + "".join(f'<span>{esc(token)}</span>' for token in tokens) + '</div>'
        if tokens is not None
        else '<span>Không khả dụng trong phiên bản Keras hiện tại.</span>'
    )
    architecture = (
        '<div class="architecture"><span>Input</span><b>→</b><span>TextVectorization</span><b>→</b>'
        '<span>Embedding</span><b>→</b><span>SimpleRNN</span><b>→</b>'
        '<span>Dropout</span><b>→</b><span>Dense + Softmax</span></div>'
        '<small class="architecture-note">Dropout tắt khi inference (training=False).</small>'
    )
    steps = (
        timeline_step(1, "Văn bản đầu vào", "Bình luận đã chọn để phân tích.",
                      f'<div class="code-box">{esc(result["text"])}</div>'),
        timeline_step(2, "Tiền xử lý", "Đầu ra chuẩn hóa, ghép token bằng khoảng trắng.",
                      f'<div class="code-box">{esc(data["normalized"]) if data["normalized"] is not None else "Không khả dụng."}</div>'),
        timeline_step(3, "Token hóa", "Các token thực tế từ TextVectorization.", token_html),
        timeline_step(4, "Chuyển token thành ID", "Tra từ vựng của checkpoint.",
                      f'<div class="code-box mono">{esc(display_ids(data["raw_ids"]))}</div>'),
        timeline_step(5, "Padding / Vectorization", "Chuỗi ID sau cắt và đệm đúng độ dài mô hình.",
                      f'<div class="code-box mono">{esc(display_ids(data["padded_ids"]))}</div>'),
        timeline_step(6, "Xử lý bằng SimpleRNN", "Kiến trúc suy luận của checkpoint.", architecture),
        timeline_step(7, "Softmax", "Ba xác suất mô hình trả về cho câu này.",
                      '<div class="compact-bars">' + "".join(bar(i, p, True) for i, p in enumerate(probabilities)) + '</div>'),
        timeline_step(8, "Kết luận", "Lớp có xác suất cao nhất.",
                      f'<div class="conclusion sentiment-bg-{index}"><strong>{LABELS_EN[index]} · {LABELS_VI[index]}</strong>'
                      f'<span>{pct(probabilities[index])}</span></div>'),
    )
    st.markdown(
        '<div class="dashboard-card timeline-card">'
        + heading("settings", "Quy trình xử lý End-to-End",
                  "Các bước xử lý văn bản và dự đoán cảm xúc từ chính câu đầu vào.", "mint")
        + '<div class="timeline">' + "".join(steps) + '</div></div>',
        unsafe_allow_html=True,
    )


def set_example(text: str) -> None:
    st.session_state.comment_text = text
    st.session_state.run_example = True


def clear_result() -> None:
    st.session_state.analysis_result = None


st.set_page_config(
    page_title="Phân tích cảm xúc tiếng Việt | HCMUTE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

model = None
model_error = None
metadata = None
if MODEL_PATH.is_file():
    try:
        model = load_sentiment_model()
        metadata = get_metadata(model)
    except Exception as exc:
        model_error = exc
else:
    model_error = FileNotFoundError(f"Không tìm thấy checkpoint: {MODEL_PATH}")

logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii") if LOGO_PATH.is_file() else None
logo_html = (
    f'<img class="school-logo" src="data:image/png;base64,{logo_data}" alt="Logo HCMUTE">'
    if logo_data else '<span class="school-logo-fallback">HCMUTE</span>'
)
status = (
    '<span class="status-pill ready"><i></i>Demo sẵn sàng</span>'
    if model is not None else '<span class="status-pill failed"><i></i>Không tải được mô hình</span>'
)
st.markdown(
    '<header class="school-header"><div class="school-brand">'
    + logo_html
    + '<div><strong>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT<br>THÀNH PHỐ HỒ CHÍ MINH</strong>'
    + '<span>Tiểu luận học phần Deep Learning</span></div></div>'
    + status + '</header>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><div class="eyebrow">BÁO CÁO MÔN HỌC <span>·</span> DEEP LEARNING '
    '<span>·</span> RECURRENT NEURAL NETWORK</div>'
    '<h1>XÂY DỰNG HỆ THỐNG PHÂN TÍCH CẢM XÚC BÌNH LUẬN TIẾNG VIỆT SỬ DỤNG RNN</h1>'
    '<p>Ứng dụng mạng nơ-ron hồi quy để phân loại cảm xúc phản hồi sinh viên tiếng Việt bằng mô hình SimpleRNN.</p>'
    '</div>',
    unsafe_allow_html=True,
)
if model is None:
    st.error("Không nạp được mô hình. Kiểm tra checkpoint và môi trường TensorFlow/Keras.")
    st.exception(model_error)
    st.stop()

team_rows = "".join(
    f'<div class="team-member"><span class="member-dot">●</span><strong>{esc(student_id)}</strong>'
    f'<span>{esc(name)}</span></div>' for student_id, name in TEAM
)
team_card = ('<div class="dashboard-card info-card">' + heading("users", "Nhóm thực hiện")
             + f'<div class="team-list">{team_rows}</div></div>')
report_card = (
    '<div class="dashboard-card info-card">' + heading("file", "Thông tin báo cáo")
    + '<div class="info-list">'
    + row("Học phần", "Deep Learning")
    + row("Chủ đề", "Phân tích cảm xúc tiếng Việt")
    + row("Mô hình", "SimpleRNN")
    + row("Số lớp cảm xúc", metadata["classes"]) + '</div></div>'
)
config_card = (
    '<div class="dashboard-card info-card">' + heading("settings", "Cấu hình mô hình", tone="mint")
    + '<div class="info-list">'
    + row("Dataset", "UIT-VSFC") + row("Ngôn ngữ", "Vietnamese")
    + row("Embedding", metadata["embedding_dim"]) + row("SimpleRNN units", metadata["rnn_units"])
    + row("Dropout", metadata["dropout"]) + row("Vocabulary size", f'{metadata["vocabulary_size"]:,}')
    + row("Sequence length", metadata["sequence_length"])
    + row("Output classes", metadata["classes"]) + '</div></div>'
)
st.markdown(f'<div class="info-grid">{team_card}{report_card}{config_card}</div>', unsafe_allow_html=True)

if "comment_text" not in st.session_state:
    st.session_state.comment_text = EXAMPLES[0]
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = run_inference(model, metadata, EXAMPLES[0])

with st.container(key="input_card"):
    st.markdown(
        heading("edit", "Nhập bình luận cần phân tích",
                "Nhập phản hồi sinh viên tiếng Việt để dự đoán cảm xúc bằng checkpoint đã huấn luyện."),
        unsafe_allow_html=True,
    )
    st.text_area(
        "Bình luận cần phân tích", key="comment_text", label_visibility="collapsed",
        placeholder="Ví dụ: Thầy giảng rất dễ hiểu.", height=104, on_change=clear_result,
    )
    analyze_clicked = st.button("Phân tích cảm xúc", type="primary", key="analyze_button")
    st.markdown('<p class="example-caption">Hoặc thử với các câu ví dụ:</p>', unsafe_allow_html=True)
    for index, (column, example) in enumerate(zip(st.columns(3), EXAMPLES)):
        with column:
            st.button(example, key=f"example_{index}", on_click=set_example,
                      args=(example,), use_container_width=True)
    if analyze_clicked or st.session_state.pop("run_example", False):
        comment = st.session_state.comment_text.strip()
        if not comment:
            st.session_state.analysis_result = None
            st.warning("Vui lòng nhập một bình luận trước khi phân tích.")
        else:
            try:
                st.session_state.analysis_result = run_inference(model, metadata, comment)
            except Exception as exc:
                st.session_state.analysis_result = None
                st.error("Có lỗi khi dự đoán. Hãy kiểm tra mô hình và thử lại.")
                st.exception(exc)

result = st.session_state.analysis_result
if result is not None and result["text"] == st.session_state.comment_text.strip():
    render_results(result, metadata)
else:
    st.markdown(
        '<div class="dashboard-card awaiting-card">'
        + heading("chart", "Kết quả dự đoán", "Nhập bình luận rồi chọn Phân tích cảm xúc để xem kết quả.")
        + '</div>', unsafe_allow_html=True,
    )
    result = None

render_stats(metadata, result)
if result is not None:
    render_timeline(result)
else:
    st.markdown(
        '<div class="dashboard-card awaiting-card">'
        + heading("settings", "Quy trình xử lý End-to-End",
                  "Timeline sẽ hiển thị dữ liệu thật sau khi phân tích.", "mint")
        + '</div>', unsafe_allow_html=True,
    )

st.markdown(
    '<footer class="page-footer"><div class="footer-brand">' + logo_html
    + '<div><strong>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH</strong>'
    + '<span>Ứng dụng phục vụ đồ án cuối kỳ Deep Learning</span></div></div>'
    + '<span>Nhóm Trịnh Nguyễn Anh Hào · Lê Huy Huân · Nguyễn Nam Triều Tiên</span>'
    + '</footer>',
    unsafe_allow_html=True,
)
