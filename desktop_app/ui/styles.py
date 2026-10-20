"""Styles and theme configuration for HCMUTE RNN Sentiment Desktop Application.

Design System:
    - Navy:           #123B7A
    - Primary Blue:   #1976D2
    - Background:     #F3F7FC
    - Card BG:        #FFFFFF
    - Border:         #DDE7F3
    - Text:           #172E52
    - Secondary Text: #59708F
    - Positive:       #10B981
    - Negative:       #EF4444
    - Neutral:        #3B82F6
"""

# Palette Constants
COLOR_NAVY = "#123B7A"
COLOR_PRIMARY = "#1976D2"
COLOR_BG = "#F3F7FC"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#DDE7F3"
COLOR_TEXT = "#172E52"
COLOR_MUTED = "#59708F"

COLOR_POSITIVE = "#10B981"
COLOR_POSITIVE_BG = "#EAF7EE"
COLOR_POSITIVE_BORDER = "#A7F3D0"

COLOR_NEUTRAL = "#3B82F6"
COLOR_NEUTRAL_BG = "#EFF6FF"
COLOR_NEUTRAL_BORDER = "#BFDBFE"

COLOR_NEGATIVE = "#EF4444"
COLOR_NEGATIVE_BG = "#FEF2F2"
COLOR_NEGATIVE_BORDER = "#FECACA"

COLOR_AWAITING = "#59708F"
COLOR_AWAITING_BG = "#F8FAFC"
COLOR_AWAITING_BORDER = "#DDE7F3"

APP_QSS = f"""
/* Global Application Style */
QWidget {{
    font-family: "Segoe UI", "Segoe UI Variable Display", Arial, sans-serif;
    color: {COLOR_TEXT};
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {COLOR_BG};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    border: none;
    background: #E8EEF5;
    width: 7px;
    border-radius: 3.5px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: #A0B2C6;
    min-height: 24px;
    border-radius: 3.5px;
}}

QScrollBar::handle:vertical:hover {{
    background: #7E93AC;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Card Styles */
QFrame.dashboard-card {{
    background-color: {COLOR_CARD};
    border-radius: 14px;
    border: 1px solid {COLOR_BORDER};
}}

/* Text Input Area */
QTextEdit#commentInput {{
    background-color: {COLOR_CARD};
    border: 1.5px solid {COLOR_BORDER};
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13.5px;
    color: {COLOR_TEXT};
    line-height: 1.4;
}}

QTextEdit#commentInput:focus {{
    border: 1.5px solid {COLOR_PRIMARY};
}}

/* Example Pill Buttons */
QPushButton.example-pill {{
    background-color: #F1F6FB;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
}}

QPushButton.example-pill:hover {{
    background-color: #E5EFF9;
    border-color: {COLOR_PRIMARY};
    color: {COLOR_NAVY};
}}

QPushButton.example-pill:pressed {{
    background-color: #D3E4F5;
}}

/* Remark Frame */
QFrame#remarkFrame {{
    background-color: #F0F6FD;
    border: 1px solid #CDE0F5;
    border-radius: 10px;
}}

QFrame#remarkFrame QLabel {{
    border: none;
    background: transparent;
}}
"""
