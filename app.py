from base64 import b64encode
from csv import DictReader
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import dedent

from PIL import Image, UnidentifiedImageError
import streamlit as st

from rice_inference import load_predictor, preprocess, read_metadata, scores


PROJECT_DIR = Path(__file__).resolve().parent
LOGO_PATH = PROJECT_DIR / "Golden rice detective icon.png"
SAMPLE_DIR = PROJECT_DIR / "Testing Samples"


@st.cache_data(show_spinner=False)
def load_logo(path):
    if not path.is_file():
        return b""
    with Image.open(path) as image:
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


logo = load_logo(LOGO_PATH)
st.set_page_config(
    page_title="Detective Rice!",
    page_icon=Image.open(BytesIO(logo)) if logo else ":material/search:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f6f1e8;
        --surface-solid: #fffdf9;
        --line: rgba(74, 57, 42, 0.10);
        --text: #2a2118;
        --muted: #6f6254;
        --accent: #2f6b4f;
        --control-ink: #183d2c;
        --control-muted: #425d50;
        --control-border: #788e80;
        --control-fill: #f1f7f2;
        --control-selected: #dceee0;
        --shadow: 0 18px 50px rgba(58, 43, 28, 0.07);
        --radius: 24px;
    }

    .stApp {
        background:
            radial-gradient(1100px 480px at 8% -10%, rgba(184, 137, 61, 0.13), transparent 60%),
            radial-gradient(900px 420px at 92% 0%, rgba(47, 107, 79, 0.11), transparent 55%),
            linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
        color: var(--text);
    }

    html, body, .stApp, .stMarkdown, .stText, p, label {
        font-family: 'Manrope', sans-serif;
    }

    .stApp .hero-title, .stApp .panel-title, .result-name, .empty-title, .hero-count {
        font-family: 'Fraunces', serif;
        letter-spacing: -0.02em;
    }

    .block-container {
        max-width: 1360px;
        padding-inline: clamp(1rem, 3vw, 2.5rem);
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Brand header */
    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: clamp(1rem, 3vw, 2rem);
        background:
            radial-gradient(ellipse at top left, #f8e9c9, transparent 58%),
            linear-gradient(115deg, #fffaf0, #eef5ec);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: clamp(1rem, 3vw, 2.1rem);
        box-shadow: var(--shadow);
    }

    .hero-brand {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        align-items: center;
        gap: clamp(0.75rem, 2vw, 1.5rem);
    }

    .hero-brand:not(:has(.hero-logo)) {
        grid-template-columns: minmax(0, 1fr);
    }

    .hero-logo {
        display: block;
        width: clamp(3.75rem, 10vw, 7rem);
        aspect-ratio: 1;
        object-fit: contain;
        background: rgba(255, 253, 249, 0.75);
        border: 1px solid rgba(136, 97, 38, 0.18);
        border-radius: clamp(14px, 2vw, 24px);
    }

    .stApp .hero-title {
        padding: 0;
        font-size: clamp(2rem, 5.1vw, 4.5rem);
        line-height: 1.02;
        margin: 0;
        color: var(--control-ink);
        font-weight: 700;
        text-wrap: balance;
    }

    .hero-title span {
        color: var(--accent);
        font-style: italic;
    }

    .hero .hero-copy {
        font-size: 0.95rem;
        line-height: 1.55;
        color: var(--muted);
        max-width: 36rem;
        margin: 0.65rem 0 0;
    }

    .hero-stat {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.65rem 1rem;
        border-radius: 16px;
        background: var(--control-ink);
        color: #fffdf9;
        font-size: 0.82rem;
        line-height: 1.35;
        font-weight: 700;
    }

    .hero-count {
        font-size: 2.4rem;
        line-height: 1;
        font-variant-numeric: tabular-nums;
        color: #f8d78e;
    }

    /* Workspace panels */
    .st-key-rice-workspace [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0;
    }

    .st-key-rice-upload, .st-key-rice-results {
        background: var(--surface-solid);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        border-radius: var(--radius);
        padding: clamp(1rem, 2.4vw, 1.5rem);
    }

    .panel-heading {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.85rem;
    }

    .stApp .panel-title {
        padding: 0;
        font-size: 1.45rem;
        line-height: 1.15;
        margin: 0 0 0.35rem;
        color: #241b13;
        font-weight: 600;
    }

    .panel-sub {
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.65;
        margin: 0;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0;
    }

    /* Give native widgets explicit colors on the app's light surfaces. */
    .st-key-rice-upload [data-testid="stWidgetLabel"],
    .st-key-rice-upload [data-testid="stWidgetLabel"] p {
        color: var(--control-ink);
        font-size: 0.9rem;
        font-weight: 800;
    }

    .st-key-rice-upload [role="radiogroup"] {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr));
        gap: 0.55rem;
    }

    .st-key-rice-upload [role="radiogroup"] label {
        display: flex;
        align-items: center;
        min-width: 0;
        min-height: 44px;
        margin: 0;
        padding: 0.62rem 0.7rem;
        color: var(--control-ink);
        background: var(--surface-solid);
        border: 1px solid var(--control-border);
        border-radius: 12px;
        cursor: pointer;
    }

    .st-key-rice-upload [role="radiogroup"] label :is(p, span) {
        color: var(--control-ink);
        font-size: 0.88rem;
        font-weight: 700;
    }

    .st-key-rice-upload [role="radiogroup"] label:hover {
        background: var(--control-fill);
    }

    .st-key-rice-upload [role="radiogroup"] label:has(input:checked) {
        background: var(--control-selected);
        border-color: var(--accent);
        box-shadow: inset 0 0 0 1px var(--accent);
    }

    .st-key-rice-upload [role="radiogroup"] label:has(input:focus-visible),
    .st-key-rice-upload button:focus-visible {
        outline: 2px solid var(--control-ink);
        outline-offset: 3px;
    }

    /* File uploader */
    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] {
        background: var(--control-fill);
        color: var(--control-ink);
        border: 1px dashed var(--control-border);
        border-radius: 16px;
        padding: 1rem;
        gap: 0.85rem;
        flex-wrap: wrap;
    }

    .st-key-rice-upload [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
        background: var(--control-selected);
    }

    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] :is(p, span) {
        color: var(--control-ink);
    }

    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] :is(small, svg),
    .st-key-rice-upload [data-testid="stFileUploaderFile"] small {
        color: var(--control-muted);
    }

    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] button {
        background: var(--control-ink);
        border: 1px solid var(--control-ink);
        border-radius: 10px;
        min-height: 44px;
    }

    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] button,
    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] button :is(p, span, svg) {
        color: #fffdf9;
        font-weight: 700;
    }

    .st-key-rice-upload [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--accent);
        border-color: var(--accent);
    }

    .st-key-rice-upload [data-testid="stFileUploaderFile"] {
        color: var(--control-ink);
        background: var(--control-fill);
        border-radius: 12px;
        border: 1px solid var(--line);
    }

    .st-key-rice-upload [data-testid="stFileUploaderFile"] :is(button, svg, span, p),
    .st-key-rice-upload [data-testid="stFileUploaderFileName"] {
        color: var(--control-ink);
    }

    .st-key-rice-upload [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height: 44px;
        background: var(--surface-solid);
        border-color: var(--control-border);
        border-radius: 12px;
    }

    .st-key-rice-upload [data-testid="stSelectbox"] [data-baseweb="select"] :is(div, span, input, svg) {
        color: var(--control-ink);
    }

    .st-key-rice-upload [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px var(--control-selected);
    }

    /* Variety chips */
    .variety-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr));
        gap: 0.65rem;
        margin: 0;
    }

    .variety-chip {
        padding: 0.8rem 0.85rem;
        border-radius: 16px;
        background: #faf6ed;
        border: 1px solid var(--line);
    }

    .variety-en {
        font-size: 0.83rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.3;
    }

    .variety-bn {
        font-size: 0.81rem;
        color: var(--muted);
        margin-top: 0.18rem;
        line-height: 1.3;
    }

    /* Results */
    .result-list {
        display: grid;
        gap: 0.65rem;
    }

    .result-card {
        padding: 0.75rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #faf7f0;
    }

    .result-card.primary {
        border-color: #b0c8b4;
        background: var(--control-fill);
    }

    .result-top {
        display: flex;
        justify-content: space-between;
        gap: 0.65rem;
        align-items: baseline;
        margin-bottom: 0.6rem;
    }

    .result-top > :first-child {
        min-width: 0;
    }

    .result-rank {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.22rem;
    }

    .result-name {
        font-size: 1.28rem;
        line-height: 1.2;
        color: #241b13;
        margin: 0;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .result-bn {
        font-size: 0.88rem;
        color: var(--muted);
        margin-top: 0.18rem;
        line-height: 1.3;
    }

    .result-score {
        font-size: 1.45rem;
        font-weight: 800;
        color: #241b13;
        font-variant-numeric: tabular-nums;
        flex: none;
        white-space: nowrap;
        line-height: 1;
    }

    .result-bar {
        width: 100%;
        height: 9px;
        background: rgba(74, 57, 42, 0.08);
        border-radius: 999px;
        overflow: hidden;
    }

    .result-bar > span {
        display: block;
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2f6b4f, #619872);
    }

    .result-card.primary .result-bar > span {
        background: linear-gradient(90deg, #2f6b4f, #b8893d);
    }

    .result-card.primary .result-name {
        font-size: 1.5rem;
    }

    .result-card.primary .result-score {
        font-size: 1.65rem;
    }

    /* Let the results panel's own width determine when the preview stacks. */
    .st-key-rice-result-detail {
        container: rice-result-detail / inline-size;
    }

    @container rice-result-detail (max-width: 30rem) {
        .st-key-rice-result-detail [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }

        .st-key-rice-result-detail [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width: 100%;
            flex: 1 1 auto;
        }
    }

    .st-key-rice-preview {
        max-width: 320px;
        margin-inline: auto;
    }

    .st-key-rice-preview [data-testid="stImage"] img {
        display: block;
        max-width: 100%;
        width: 100%;
        height: auto;
        object-fit: contain;
        border-radius: 12px;
        background: #171713;
    }

    .st-key-rice-preview [data-testid="stImageCaption"],
    .st-key-rice-preview [data-testid="stImageCaption"] p {
        color: var(--muted);
    }

    /* Empty state */
    .empty-state {
        padding: 2.75rem 1.5rem;
        text-align: center;
        color: var(--muted);
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(135deg, #f0f6ef, #faf6ed);
    }

    .empty-icon {
        width: 52px;
        height: 52px;
        margin: 0 auto 0.85rem;
        border-radius: 16px;
        background: rgba(47, 107, 79, 0.1);
        color: var(--accent);
        display: grid;
        place-items: center;
    }

    .empty-icon svg {
        width: 26px;
        height: 26px;
        display: block;
    }

    .empty-title {
        font-size: 1.4rem;
        color: var(--text);
        margin: 0 0 0.4rem;
        font-weight: 600;
    }

    .empty-copy {
        font-size: 0.92rem;
        line-height: 1.65;
        max-width: 28rem;
        margin: 0 auto;
    }

    .footnote {
        padding: 0.95rem 1.05rem;
        border-radius: 16px;
        background: rgba(184, 137, 61, 0.08);
        border: 1px solid rgba(184, 137, 61, 0.14);
        color: #5e4f3d;
        font-size: 0.86rem;
        line-height: 1.65;
    }

    .sample-reference {
        margin: 0;
        padding: 0.65rem 0.8rem;
        background: var(--control-fill);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        color: var(--control-ink);
        font-size: 0.88rem;
    }

    .footnote strong {
        color: #4a3b2a;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    @media (max-width: 960px) {
        .st-key-rice-workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-rice-upload) {
            flex-direction: column;
        }
        .st-key-rice-workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-rice-upload) > [data-testid="stColumn"] {
            width: 100%;
            flex: 1 1 auto;
        }
    }

    @media (max-width: 700px) {
        .hero {
            grid-template-columns: 1fr;
            gap: 0.85rem;
        }

        .hero-stat {
            justify-self: start;
        }
    }

    @media (max-width: 600px) {
        .block-container {
            padding-inline: 1rem;
            padding-bottom: 2rem;
        }
        .empty-state { padding: 1.5rem 1rem; }
        .result-card.primary .result-score { font-size: 1.45rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

metadata = read_metadata()
class_count = len(metadata["classes"])
class_names = {item["dataset_id"]: item["name_en"] for item in metadata["classes"]}
samples = {}
sample_counts = {}
manifest = SAMPLE_DIR / "labels.csv"
if manifest.is_file():
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        for row in DictReader(file):
            filename = row["filename"]
            class_id = row["class_id"]
            sample_path = SAMPLE_DIR / filename
            if class_id in class_names and Path(filename).name == filename and sample_path.is_file():
                number = sample_counts.get(class_id, 0) + 1
                sample_counts[class_id] = number
                samples[f"{class_names[class_id]} · Photo {number}"] = (sample_path, class_names[class_id])

logo_markup = (
    f'<img class="hero-logo" src="data:image/png;base64,{b64encode(logo).decode("ascii")}" alt="">'
    if logo else ""
)


@st.cache_resource
def cached_predictor(selected_backend):
    return load_predictor(selected_backend, read_metadata())


st.markdown(
    f"""
    <div class="hero">
        <div class="hero-brand">
            {logo_markup}
            <div>
                <h1 class="hero-title">Detective <span>Rice!</span></h1>
                <p class="hero-copy">
                    Photograph one uncooked, dehusked grain up close on a matte black background.
                </p>
            </div>
        </div>
        <div class="hero-stat">
            <span class="hero-count">{class_count}</span>
            <span>Bangladeshi<br>rice varieties</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(key="rice-workspace"):
    left_col, right_col = st.columns([0.78, 1.62], gap="medium")

sample = None
with left_col, st.container(key="rice-upload"):
    st.markdown(
        '<div class="panel-heading"><h2 class="panel-title">Analyze a grain</h2>'
        '<p class="panel-sub">Upload a photo or choose a test sample.</p></div>',
        unsafe_allow_html=True,
    )

    backend = st.radio("Inference engine", ["ONNX Runtime", "PyTorch"], horizontal=True)
    upload = st.file_uploader(
        "Grain photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )

    if samples:
        selection = st.selectbox(
            "Sample photo",
            options=list(samples),
            index=None,
            placeholder="Choose a held-out test photo",
            disabled=upload is not None,
        )
        if selection is not None and upload is None:
            sample = samples[selection]

    st.markdown('<div class="section-label">Supported varieties</div>', unsafe_allow_html=True)
    chips = []
    for label in metadata["classes"]:
        chips.append(
            '<div class="variety-chip">'
            f'<div class="variety-en">{label["name_en"]}</div>'
            f'<div class="variety-bn">{label["name_bn"]}</div>'
            "</div>"
        )
    st.markdown(f'<div class="variety-grid">{"".join(chips)}</div>', unsafe_allow_html=True)

with right_col, st.container(key="rice-results"):
    st.markdown('<h2 class="panel-title">Results</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-sub">Top matches are ranked by model score. The preview shows the exact 224 × 224 image fed to the model.</p>',
        unsafe_allow_html=True,
    )

    if upload is None and sample is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="6.25"></circle>
                        <path d="M16 16l4.25 4.25"></path>
                    </svg>
                </div>
                <p class="empty-title">No photo selected yet</p>
                <p class="empty-copy">Choose a photo to see the model input and the three closest variety matches.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image_source = sample[0] if sample is not None else upload
            with Image.open(image_source) as photo:
                tensor, preview = preprocess(photo, metadata)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("Could not open the image. Try a standard JPEG or PNG.")
            st.stop()

        try:
            probabilities = scores(cached_predictor(backend)(tensor[None])[0])
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not run {backend}. {exc}")
            st.stop()

        if sample is not None:
            st.markdown(
                f'<p class="sample-reference">Test-set label <strong>{escape(sample[1])}</strong></p>',
                unsafe_allow_html=True,
            )

        with st.container(key="rice-result-detail"):
            results_col, preview_col = st.columns([1.1, 1], gap="medium")
        with results_col:
            cards = []
            ranked = probabilities.argsort()[::-1][:3]
            labels = ["Top match", "Second match", "Third match"]
            for rank, index in enumerate(ranked):
                label = metadata["classes"][index]
                probability = float(probabilities[index])
                pct = probability * 100
                primary = " primary" if rank == 0 else ""
                cards.append(dedent(
                    f"""
                    <div class="result-card{primary}">
                        <div class="result-top">
                            <div>
                                <div class="result-rank">{labels[rank]}</div>
                                <div class="result-name">{label["name_en"]}</div>
                                <div class="result-bn">{label["name_bn"]}</div>
                            </div>
                            <div class="result-score">{pct:.1f}%</div>
                        </div>
                        <div class="result-bar"><span style="width:{pct:.2f}%"></span></div>
                    </div>
                    """
                ).strip())

            st.markdown(f'<div class="result-list">{"".join(cards)}</div>', unsafe_allow_html=True)

        with preview_col, st.container(key="rice-preview"):
            buf = BytesIO()
            preview.save(buf, format="PNG")
            width, height = preview.size
            st.image(buf.getvalue(), width=320, caption=f"Model input ({width} × {height} RGB)")

        st.markdown(
            '<div class="footnote"><strong>Note:</strong> Trained on microscope images. Scores compare the six supported varieties. Fresh phone-photo performance is unmeasured.</div>',
            unsafe_allow_html=True,
        )
