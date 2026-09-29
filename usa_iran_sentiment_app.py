"""
USA-Iran Conflict Sentiment Classifier — Streamlit App
--------------------------------------------------------
TF-IDF + Multinomial Naive Bayes sentiment classifier trained on
MAYOWA_DATASET.csv (18,019 cleaned social media comments about the
USA-Iran conflict, labelled positive/negative/neutral).

The model deployed here is trained on the FULL cleaned dataset (to give
the live app the most data possible); a separate, honest 80/20 held-out
split (identical to the one used in the evaluation notebook) is run once
at startup purely to report a real accuracy figure in the UI, never to
influence the deployed model itself.

UI is a chat-thread layout inspired by ChatGPT's interface (dark
sidebar with session history, centered thread, fixed bottom composer),
following the same pattern as andrew_bed_occupancy_app_v2.py but with a
free-text composer instead of a structured numeric form — no OpenAI
branding, logos, or assets are used.

Run:
    streamlit run /Users/user/anaconda_projects/usa_iran_sentiment_app.py
"""

import os
import re
import string
import time as time_module
from collections import Counter

import nltk
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "MAYOWA_DATASET.csv")

NEGATION_WORDS = {"not", "no", "never"}
RARE_WORD_THRESHOLD = 1


def html(s):
    """Render a multi-line HTML string via st.markdown.

    Two Streamlit markdown quirks are worked around here:
    1. Leading whitespace on a line is treated as a Markdown code
       block, so nested `with`/`if` blocks (whose Python indentation
       leaks into the string) would otherwise render as literal text.
    2. A blank line inside a large raw-HTML block (e.g. a long
       <style> block) makes the parser drop out of "raw HTML" mode
       partway through, leaking the remainder as visible text — so
       blank lines are stripped entirely, not just dedented.
    """
    cleaned = "\n".join(line.strip() for line in s.strip("\n").splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ICONS — inline SVG (Lucide-style outline icons), no emoji anywhere in the UI
# ─────────────────────────────────────────────────────────────────────────────

ICONS = {
    "message": '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><circle cx="12" cy="8" r="0.75" fill="currentColor" stroke="none"/>',
    "arrow-up": '<path d="M12 19V5"/><path d="M5 12l7-7 7 7"/>',
}


def icon(name, size=16, stroke="currentColor", stroke_width=2):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;">{ICONS[name]}</svg>'
    )


def icon_data_uri(name, color="black", stroke_width=2.2):
    import urllib.parse

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def button_icon_css(container_key, icon_name):
    """CSS to prepend an SVG icon to a native st.button via mask-image,
    since st.button's label can't contain raw HTML/SVG."""
    uri = icon_data_uri(icon_name)
    return f"""
    .st-key-{container_key} button::before {{
        content: "";
        display: inline-block; width: 15px; height: 15px;
        background-color: currentColor;
        -webkit-mask: url('{uri}') no-repeat center / contain;
        mask: url('{uri}') no-repeat center / contain;
        margin-right: 8px; vertical-align: -3px;
    }}
    """


# ─────────────────────────────────────────────────────────────────────────────
# FAVICON — generated to match the in-app message icon (no emoji favicon either)
# ─────────────────────────────────────────────────────────────────────────────

FAVICON_PATH = os.path.join(BASE_DIR, ".sentiment_favicon.png")
if not os.path.exists(FAVICON_PATH):
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=16, fill=(43, 53, 97, 255))
    draw.rounded_rectangle([12, 14, 52, 40], radius=9, outline="white", width=3)
    draw.polygon([(20, 40), (20, 50), (30, 40)], fill="white")
    for cx in (24, 32, 40):
        draw.ellipse([cx - 2.5, 25, cx + 2.5, 30], fill="white")
    img.save(FAVICON_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="USA-Iran Conflict Sentiment Classifier", page_icon=FAVICON_PATH, layout="wide")

if "history" not in st.session_state:
    st.session_state["history"] = []
if "pending" not in st.session_state:
    st.session_state["pending"] = None

# Dark theme only (matching the companion hospital bed occupancy app).
T = {
    "bg_main": "#212121",
    "bg_sidebar": "#171717",
    "bg_secondary": "#2F2F2F",
    "border": "#3A3A3A",
    "text_primary": "#ECECEC",
    "text_secondary": "#B4B4B4",
    "text_tertiary": "#8E8EA0",
    "bubble_user": "#2F2F2F",
    "accent_bg": "#FFFFFF",
    "accent_fg": "#0D0D0D",
    "hover": "rgba(255,255,255,0.06)",
    "gauge_track": "#3A3A3A",
    "avatar_bg": "linear-gradient(135deg, #4C51BF 0%, #2D3561 100%)",
}

# Sentiment colors kept consistent with the evaluation notebook's own charts
# (Figure 4.3, Table 4.9's grouped bar chart), so the app and the document
# visually agree with each other.
SENTIMENT_STYLE = {
    "positive": {"arc": "#55A868", "bg": "rgba(85,168,104,0.16)", "fg": "#7FD996", "border": "rgba(85,168,104,0.35)"},
    "negative": {"arc": "#C44E52", "bg": "rgba(196,78,82,0.16)", "fg": "#F08488", "border": "rgba(196,78,82,0.35)"},
    "neutral": {"arc": "#4C72B0", "bg": "rgba(76,114,176,0.16)", "fg": "#8FB2E8", "border": "rgba(76,114,176,0.35)"},
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

html(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>

    html {{ scroll-behavior: smooth; }}
    html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4, button, input, textarea {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent !important; box-shadow: none !important; }}
    .stApp {{ background-color: {T['bg_main']}; transition: background-color 0.25s ease; }}
    .tnum {{ font-variant-numeric: tabular-nums; }}

    /* ── Chat layout: the thread scrolls in its own bounded pane, the
       composer sits below it as a normal (non-floating) flex sibling. ── */
    [data-testid="stMain"] {{
        height: 100vh !important; overflow: hidden !important;
        display: flex !important; flex-direction: column !important;
    }}
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
        display: flex !important; flex-direction: column !important;
        height: 100% !important; overflow: hidden !important;
        padding-top: 0.6rem !important; padding-bottom: 0.6rem !important;
        max-width: none !important;
    }}
    [data-testid="stLayoutWrapper"]:has(.st-key-thread_scroll) {{
        flex: 1 1 auto !important; min-height: 0 !important; overflow: hidden !important;
    }}
    [data-testid="stLayoutWrapper"]:has(.st-key-composer) {{ flex-shrink: 0 !important; }}
    .st-key-thread_scroll {{ height: 100% !important; overflow-y: auto !important; }}
    .st-key-thread_scroll > div {{ max-width: 780px; margin: 0 auto; padding: 0 1.5rem; }}

    .material-icons, .material-icons-round, .material-icons-outlined,
    .material-symbols-rounded, .material-symbols-outlined, [data-testid="stIconMaterial"] {{
        font-size: 0 !important; width: 0 !important; overflow: hidden !important;
    }}

    /* ── Sidebar ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {T['bg_sidebar']} !important;
        border-right: 1px solid {T['border']} !important;
        transition: background-color 0.25s ease;
    }}
    [data-testid="stSidebar"] * {{ color: {T['text_primary']}; }}
    [data-testid="stSidebarHeader"] {{ padding: 10px 12px !important; }}

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stExpandSidebarButton"] {{
        width: 32px !important; height: 32px !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        border-radius: 8px !important; border: none !important; cursor: pointer !important;
        position: relative !important; padding: 0 !important;
        transition: background 0.15s ease !important;
    }}
    [data-testid="stSidebarCollapseButton"] button {{
        background: transparent !important; color: {T['text_secondary']} !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{ background: {T['hover']} !important; }}
    [data-testid="stExpandSidebarButton"] {{
        background: {T['bg_secondary']} !important; border: 1px solid {T['border']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    }}
    [data-testid="stExpandSidebarButton"]:hover {{ background: {T['hover']} !important; }}
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{ display: none !important; }}
    [data-testid="stSidebarCollapseButton"] button::before,
    [data-testid="stExpandSidebarButton"]::before {{
        content: ""; position: absolute; width: 16px; height: 2px; border-radius: 2px;
        background: currentColor; top: calc(50% - 6px); left: calc(50% - 8px);
        box-shadow: 0 5px 0 currentColor, 0 10px 0 currentColor;
    }}
    [data-testid="stExpandSidebarButton"]::before {{ color: {T['text_primary']}; }}

    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important; border: 1px solid {T['border']} !important;
        border-radius: 12px !important; color: {T['text_primary']} !important;
        font-size: 13.5px !important; font-weight: 600 !important;
        padding: 10px 14px !important; width: 100% !important; text-align: left !important;
        transition: background 0.15s ease !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{ background: {T['hover']} !important; }}

    .sb-section-label {{
        font-size: 11px; font-weight: 700; color: {T['text_tertiary']};
        text-transform: uppercase; letter-spacing: 0.6px; padding: 6px 14px; margin-top: 6px;
    }}
    .sb-row {{
        display: block; text-decoration: none !important;
        padding: 9px 14px; margin: 1px 8px; border-radius: 9px;
        font-size: 13px; font-weight: 500; color: {T['text_primary']} !important;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        transition: background 0.15s ease;
    }}
    .sb-row:hover {{ background: {T['hover']}; }}
    .sb-row .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px; }}
    .sb-row .meta {{ color: {T['text_secondary']}; font-size: 11.5px; margin-left: 15px; }}
    .sb-empty {{ font-size: 12px; color: {T['text_tertiary']}; padding: 8px 14px; line-height: 1.5; }}

    .sb-model-card {{
        margin: 4px 12px 14px 12px; padding: 10px 12px;
        border: 1px solid {T['border']}; border-radius: 12px;
        background: {T['bg_secondary']};
    }}
    .sb-model-card .t {{ font-size: 12px; font-weight: 700; color: {T['text_primary']}; margin-bottom: 4px; }}
    .sb-model-card .d {{ font-size: 11px; color: {T['text_secondary']}; line-height: 1.5; }}

    /* ── Landing hero (empty thread) ──────────────────────────────── */
    .hero {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 12vh 20px 20px 20px; animation: fadeIn 0.4s ease;
    }}
    .hero-icon {{
        width: 56px; height: 56px; border-radius: 16px; background: {T['avatar_bg']};
        display: flex; align-items: center; justify-content: center; font-size: 27px;
        box-shadow: 0 4px 14px rgba(45,53,97,0.35); margin-bottom: 16px;
    }}
    .hero-title {{ font-size: 27px; font-weight: 800; color: {T['text_primary']}; letter-spacing: -0.5px; }}
    .hero-sub {{ font-size: 14px; color: {T['text_secondary']}; margin-top: 6px; max-width: 460px; line-height: 1.5; }}
    .hero-examples {{ margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; max-width: 560px; }}
    .hero-chip {{
        font-size: 12px; font-weight: 600; color: {T['text_secondary']};
        background: {T['bg_secondary']}; border: 1px solid {T['border']}; border-radius: 20px;
        padding: 6px 13px;
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

    /* ── Thread ────────────────────────────────────────────────────── */
    .turn {{ margin-bottom: 26px; animation: slideUp 0.32s ease; }}
    @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

    .user-row {{ display: flex; justify-content: flex-end; margin-bottom: 18px; }}
    .user-bubble {{
        background: {T['bubble_user']}; color: {T['text_primary']};
        border-radius: 18px 18px 4px 18px; padding: 11px 16px;
        font-size: 14px; line-height: 1.55; max-width: 78%; white-space: pre-wrap;
    }}

    .assistant-row {{ display: flex; gap: 12px; align-items: flex-start; }}
    .assistant-avatar {{
        width: 30px; height: 30px; border-radius: 8px; background: {T['avatar_bg']};
        display: flex; align-items: center; justify-content: center; font-size: 15px;
        flex-shrink: 0; margin-top: 2px;
    }}
    .assistant-body {{ flex: 1; min-width: 0; }}
    .assistant-name {{ font-size: 12.5px; font-weight: 700; color: {T['text_primary']}; margin-bottom: 6px; }}

    .thinking-dots {{ display: inline-flex; gap: 4px; padding: 6px 0; }}
    .thinking-dots span {{
        width: 6px; height: 6px; border-radius: 50%; background: {T['text_tertiary']};
        animation: bounce 1.2s infinite ease-in-out;
    }}
    .thinking-dots span:nth-child(2) {{ animation-delay: 0.15s; }}
    .thinking-dots span:nth-child(3) {{ animation-delay: 0.3s; }}
    @keyframes bounce {{ 0%, 60%, 100% {{ transform: translateY(0); opacity: 0.5; }} 30% {{ transform: translateY(-5px); opacity: 1; }} }}

    /* ── Result card content ──────────────────────────────────────── */
    .badge {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; border: 1px solid transparent;
    }}
    .badge-dot {{ width: 7px; height: 7px; border-radius: 50%; }}
    .confidence-strip {{
        display: flex; align-items: flex-start; gap: 9px;
        background: {T['bg_secondary']}; border: 1px solid {T['border']}; border-radius: 10px;
        padding: 10px 13px; margin-top: 14px; max-width: 480px;
    }}
    .confidence-strip .ci {{ font-size: 14px; line-height: 1.3; }}
    .confidence-strip .ct {{ font-size: 11.5px; color: {T['text_secondary']}; line-height: 1.5; font-weight: 500; }}
    .confidence-strip .ct b {{ color: {T['text_primary']}; }}

    /* ── Sentiment probability bars ───────────────────────────────── */
    .bars-wrap {{ display: flex; flex-direction: column; gap: 9px; width: 100%; max-width: 340px; margin: 4px 0 2px 0; }}
    .bar-row {{ display: flex; flex-direction: column; gap: 3px; }}
    .bar-label-row {{ display: flex; justify-content: space-between; font-size: 12px; }}
    .bar-label {{ font-weight: 600; color: {T['text_secondary']}; }}
    .bar-label.is-top {{ color: {T['text_primary']}; }}
    .bar-pct {{ font-weight: 700; color: {T['text_primary']}; }}
    .bar-track {{ height: 8px; border-radius: 4px; background: {T['gauge_track']}; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 4px; width: 0%; }}

    /* ── Composer: a normal (non-floating) flex item docked below the
       scrolling thread pane — see [data-testid="stMain"] above. ──────── */
    .st-key-composer {{
        flex-shrink: 0 !important;
        background: {T['bg_main']};
        padding: 10px 0 4px 0 !important;
    }}
    .st-key-composer > div {{ max-width: 780px; margin: 0 auto; padding: 0 1.5rem; }}
    [data-testid="stForm"] {{
        background: {T['bg_secondary']} !important; border: 1px solid {T['border']} !important;
        border-radius: 26px !important; padding: 6px 8px 6px 20px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        transition: border-color 0.15s ease !important;
    }}
    [data-testid="stForm"]:focus-within {{ border-color: {T['text_secondary']} !important; }}
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{ align-items: center !important; }}

    .stTextInput input {{
        background: transparent !important; border: none !important; box-shadow: none !important;
        font-size: 14.5px !important; color: {T['text_primary']} !important; padding: 10px 0 !important;
    }}
    .stTextInput input::placeholder {{ color: {T['text_tertiary']} !important; }}
    [data-testid="stForm"] [data-baseweb="input"],
    [data-testid="stForm"] [data-baseweb="base-input"] {{
        background: transparent !important; border: none !important; box-shadow: none !important;
    }}
    /* Streamlit's built-in "Press Enter to submit form" hint is absolutely
       positioned inside the input's own box, directly overlapping its
       tappable/clickable area at every viewport width (verified: their
       bounding boxes genuinely intersect, not just a visual near-miss),
       which silently eats clicks/taps landing on that part of the input. */
    [data-testid="stForm"] [data-testid="InputInstructions"] {{ display: none !important; }}
    [data-testid="stFormSubmitButton"] > button {{
        background-color: {T['accent_bg']} !important; color: {T['accent_fg']} !important;
        border: none !important; border-radius: 50% !important;
        width: 40px !important; height: 40px !important; min-height: 40px !important;
        padding: 0 !important; font-weight: 700 !important;
        position: relative !important; overflow: hidden !important;
        transition: transform 0.12s ease, opacity 0.15s ease !important;
    }}
    [data-testid="stFormSubmitButton"] > button p {{ font-size: 0 !important; line-height: 0 !important; }}
    [data-testid="stFormSubmitButton"] > button::before {{
        content: ""; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 18px; height: 18px; background-color: {T['accent_fg']};
        -webkit-mask: url('{icon_data_uri("arrow-up", color="black", stroke_width=2.5)}') no-repeat center / contain;
        mask: url('{icon_data_uri("arrow-up", color="black", stroke_width=2.5)}') no-repeat center / contain;
    }}
    [data-testid="stFormSubmitButton"] > button:hover {{ transform: scale(1.06) !important; }}
    [data-testid="stFormSubmitButton"] > button:active {{ transform: scale(0.96) !important; }}
    [data-testid="stFormSubmitButton"] {{ display: flex !important; justify-content: flex-end !important; }}

    /* ── Responsive (mobile) ───────────────────────────────────────── */
    html, body {{ overflow-x: hidden !important; }}

    @media (max-width: 640px) {{
        .st-key-thread_scroll > div {{ padding: 0 0.85rem; }}
        .hero {{ padding: 6vh 8px 16px 8px; }}
        .hero-icon {{ width: 44px; height: 44px; border-radius: 13px; margin-bottom: 12px; }}
        .hero-title {{ font-size: 21px; }}
        .hero-sub {{ font-size: 13px; max-width: 320px; }}

        .user-bubble {{ max-width: 88%; font-size: 13.5px; }}
        .assistant-avatar {{ width: 26px; height: 26px; }}
        .confidence-strip {{ max-width: 100%; }}
        .bars-wrap {{ max-width: 100%; }}

        .st-key-composer {{ padding: 8px 0 4px 0 !important; }}
        .st-key-composer > div {{ padding: 0 0.6rem; }}
        [data-testid="stForm"] {{ padding: 4px 6px 4px 14px !important; border-radius: 22px !important; }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
            flex: 1 1 auto !important; min-width: 0 !important; width: auto !important;
        }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
            flex: 0 0 40px !important; width: 40px !important; min-width: 40px !important;
        }}
        .stTextInput input {{ font-size: 13.5px !important; padding-right: 6px !important; }}
    }}

    </style>
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP — quiet, idempotent download of the resources preprocessing needs
# ─────────────────────────────────────────────────────────────────────────────

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                  "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — identical pipeline to the evaluation notebook (Stage 2 & 3),
# minus the frequent-word step, which the notebook found should remove nothing
# ─────────────────────────────────────────────────────────────────────────────

_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english")) - NEGATION_WORDS


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_special_characters(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in _stop_words)


def remove_words(text, word_set):
    return " ".join(w for w in text.split() if w not in word_set)


def _wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def stem_tokens(tokens):
    return [_stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    tags = nltk.pos_tag(tokens)
    return [_lemmatizer.lemmatize(w, _wordnet_pos(t)) for w, t in tags]


def preprocess_series(text_series):
    """Runs the full cleaning + normalization + stopword pipeline on a pandas
    Series, returning the corpus-wide rare-word set alongside the cleaned text
    (rare words can only be identified once the whole corpus is known)."""
    cleaned = (
        text_series.astype(str)
        .apply(remove_punctuation)
        .apply(remove_special_characters)
        .str.lower()
        .apply(remove_stopwords)
    )
    word_counts = Counter(" ".join(cleaned).split())
    rare_words = {w for w, c in word_counts.items() if c <= RARE_WORD_THRESHOLD}
    cleaned = cleaned.apply(lambda t: remove_words(t, rare_words))
    return cleaned, rare_words


def finish_pipeline(text_series):
    tokens = text_series.apply(word_tokenize)
    stemmed = tokens.apply(stem_tokens)
    lemmatized = stemmed.apply(lemmatize_tokens)
    return lemmatized.apply(lambda ts: " ".join(ts))


def preprocess_custom_text(text, rare_words):
    """Same pipeline as above, applied to a single new comment at prediction
    time, reusing the exact same rare-word set learned from training data."""
    text = remove_special_characters(remove_punctuation(str(text))).lower()
    text = remove_stopwords(text)
    text = remove_words(text, rare_words)
    tokens = word_tokenize(text)
    tokens = stem_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    data = df[["cleaned_comment", "vader_sentiment"]].copy()
    data = data.dropna(subset=["cleaned_comment", "vader_sentiment"]).reset_index(drop=True)
    data = data.drop_duplicates(subset=["cleaned_comment"]).reset_index(drop=True)

    cleaned_text, rare_words = preprocess_series(data["cleaned_comment"])
    data["processed_comment"] = finish_pipeline(cleaned_text)

    X = data["processed_comment"]
    y = data["vader_sentiment"]

    # Deployed model: trained on the FULL cleaned dataset for real-world use.
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_df=0.9, binary=True, norm="l1")
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    model = MultinomialNB(alpha=0.3, fit_prior=False)
    model.fit(X_tfidf, y)

    # Separate, honest held-out evaluation (identical split to the notebook),
    # purely to report a real accuracy figure — never used to fit the model above.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    eval_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_df=0.9, binary=True, norm="l1")
    X_train_tfidf = eval_vectorizer.fit_transform(X_train)
    X_test_tfidf = eval_vectorizer.transform(X_test)
    eval_model = MultinomialNB(alpha=0.3, fit_prior=False)
    eval_model.fit(X_train_tfidf, y_train)
    held_out_accuracy = accuracy_score(y_test, eval_model.predict(X_test_tfidf))

    return model, tfidf_vectorizer, rare_words, held_out_accuracy, len(data)


model, tfidf_vectorizer, rare_words, held_out_accuracy, dataset_size = load_and_train()
CLASS_ORDER = ["positive", "negative", "neutral"]


def predict_sentiment(text):
    processed = preprocess_custom_text(text, rare_words)
    vector = tfidf_vectorizer.transform([processed])
    predicted_label = model.predict(vector)[0]
    proba = dict(zip(model.classes_, model.predict_proba(vector)[0]))
    proba_pct = {label: proba.get(label, 0.0) * 100 for label in CLASS_ORDER}
    return predicted_label, proba_pct


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def bars_svg(proba_pct, predicted_label):
    rows = []
    for label in CLASS_ORDER:
        pct = proba_pct[label]
        style = SENTIMENT_STYLE[label]
        is_top = label == predicted_label
        label_class = "bar-label is-top" if is_top else "bar-label"
        rows.append(
            f"""
            <div class="bar-row">
            <div class="bar-label-row">
            <span class="{label_class}">{label.capitalize()}</span>
            <span class="bar-pct tnum">{pct:.1f}%</span>
            </div>
            <div class="bar-track">
            <div class="bar-fill" style="background:{style['arc']};
            animation: barfill-{abs(hash((label, round(pct, 2)))) % 100000} 0.7s ease-out forwards;"></div>
            </div>
            </div>
            <style>
            @keyframes barfill-{abs(hash((label, round(pct, 2)))) % 100000} {{ to {{ width: {pct:.2f}%; }} }}
            </style>
            """
        )
    return f'<div class="bars-wrap">{"".join(rows)}</div>'


def render_assistant_result(text, predicted_label, proba_pct):
    style = SENTIMENT_STYLE[predicted_label]
    confidence = proba_pct[predicted_label]
    badge = (
        f'<span class="badge" style="background:{style["bg"]};color:{style["fg"]};border-color:{style["border"]};">'
        f'<span class="badge-dot" style="background:{style["fg"]};"></span>{predicted_label.capitalize()} sentiment</span>'
    )
    return f"""
    <div class="assistant-row">
    <div class="assistant-avatar">{icon('message', size=15, stroke='white')}</div>
    <div class="assistant-body">
    <div class="assistant-name">Sentiment Analysis Result</div>
    <div style="margin-bottom:10px;">{badge}</div>
    {bars_svg(proba_pct, predicted_label)}
    <div class="confidence-strip">
    <div class="ci">{icon('info', size=15)}</div>
    <div class="ct">
    <b>What this means:</b> the model is {confidence:.0f}% confident this comment expresses
    <b>{predicted_label}</b> sentiment toward the USA-Iran conflict, based on patterns learned from
    {dataset_size:,} real social media comments. On comments it has never seen before, this model is
    correct about {held_out_accuracy * 100:.0f}% of the time overall.
    </div>
    </div>
    </div>
    </div>
    """


def scroll_thread_to_bottom():
    """Push the newest turn into view, the way a chat app does when a
    message arrives. Runs via a 0-height components.html iframe since
    st.markdown strips <script> tags; the iframe's script can still
    reach the real page through window.parent (same origin).

    Streamlit dedupes a component by its rendered content: identical
    HTML across reruns means the iframe is never remounted, so the
    <script> inside it never re-executes past the very first call.
    A nonce (turn count + wall-clock time) makes every call's payload
    unique so a fresh mount — and a fresh script run — actually happens
    each time a new turn arrives.
    """
    nonce = f"{len(st.session_state['history'])}-{time_module.time()}"
    components.html(
        f"""<script>
        // nonce: {nonce}
        function scrollNow() {{
            const doc = window.parent.document;
            const el = doc.querySelector('.st-key-thread_scroll');
            if (el) {{ el.scrollTop = el.scrollHeight; }}
        }}
        setTimeout(scrollNow, 150);
        setTimeout(scrollNow, 500);
        </script>""",
        height=0,
    )


def truncate(text, n=48):
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    html(f"<style>{button_icon_css('new_analysis', 'plus')}</style>")

    if st.button("New analysis", key="new_analysis", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    html('<div class="sb-section-label">History</div>')
    if st.session_state["history"]:
        rows = []
        for i, turn in enumerate(reversed(st.session_state["history"])):
            idx = len(st.session_state["history"]) - i
            style = SENTIMENT_STYLE[turn["prediction"]]
            rows.append(
                f'<a class="sb-row" href="#turn-{idx}">'
                f'<span class="dot" style="background:{style["arc"]};"></span>'
                f'{truncate(turn["text"], 34)}'
                f'<div class="meta">{turn["prediction"].capitalize()} · '
                f'{turn["proba"][turn["prediction"]]:.0f}% confidence</div></a>'
            )
        html("".join(rows))
    else:
        html('<div class="sb-empty">Your analyzed comments this session will appear here.</div>')

    html(
        f"""
        <div class="sb-model-card">
        <div class="t">Model</div>
        <div class="d">TF-IDF + Multinomial Naive Bayes<br>
        Trained on {dataset_size:,} comments<br>
        {held_out_accuracy * 100:.1f}% held-out accuracy</div>
        </div>
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN: THREAD
# ─────────────────────────────────────────────────────────────────────────────

with st.container(key="thread_scroll"):
    if not st.session_state["history"]:
        html(
            f"""
            <div class="hero">
            <div class="hero-icon">{icon('message', size=26, stroke='white')}</div>
            <div class="hero-title">USA-Iran Conflict Sentiment Classifier</div>
            <div class="hero-sub">Type a social media comment about the USA-Iran conflict below to see
            whether a TF-IDF + Multinomial Naive Bayes model, trained on {dataset_size:,} real comments,
            reads it as positive, negative, or neutral.</div>
            <div class="hero-examples">
            <span class="hero-chip">"Trump is doing the right thing"</span>
            <span class="hero-chip">"This war is not good for anyone"</span>
            <span class="hero-chip">"Iran and the US have talked before"</span>
            </div>
            </div>
            """
        )
    else:
        for i, turn in enumerate(st.session_state["history"], start=1):
            html(
                f'<div class="turn" id="turn-{i}">'
                f'<div class="user-row"><div class="user-bubble">{turn["text"]}</div></div>'
                f'{render_assistant_result(turn["text"], turn["prediction"], turn["proba"])}'
                f'</div>'
            )

    if st.session_state["pending"] is not None:
        html(
            f"""
            <div class="turn">
            <div class="user-row"><div class="user-bubble">{st.session_state["pending"]}</div></div>
            <div class="assistant-row">
            <div class="assistant-avatar">{icon('message', size=15, stroke='white')}</div>
            <div class="assistant-body">
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            </div>
            </div>
            </div>
            """
        )

        pending_text = st.session_state["pending"]
        time_module.sleep(0.45)
        predicted_label, proba_pct = predict_sentiment(pending_text)
        st.session_state["history"].append({"text": pending_text, "prediction": predicted_label, "proba": proba_pct})
        st.session_state["pending"] = None
        st.rerun()

scroll_thread_to_bottom()

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSER (fixed bottom)
# ─────────────────────────────────────────────────────────────────────────────

with st.container(key="composer"):
    with st.form("predict_form", border=False, clear_on_submit=True):
        cols = st.columns([6, 1])
        with cols[0]:
            comment_text = st.text_input(
                "Comment", placeholder="Type a comment about the USA-Iran conflict…",
                label_visibility="collapsed", key="in_comment",
            )
        with cols[1]:
            submitted = st.form_submit_button("Analyze")

if submitted and comment_text.strip():
    st.session_state["pending"] = comment_text.strip()
    st.rerun()
