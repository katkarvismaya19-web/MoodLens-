"""MoodLens - Face Emotion Recognition System.  Run:  streamlit run app.py"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import db
from emotions import EMOJI, EMOTIONS, label, positivity

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "emotion_cnn.pt"
METRICS = ROOT / "models" / "metrics.json"
GRAPHS = ROOT / "models" / "graphs"
UPLOADS = ROOT / "data" / "uploads"

st.set_page_config(page_title="MoodLens", page_icon="🎭", layout="wide")
db.init()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {font-family: 'Inter', sans-serif !important;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}

/* top navigation bar */
.st-key-navbar {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
                padding: .55rem 1rem; margin-bottom: 1.4rem;
                box-shadow: 0 1px 3px rgba(15,23,42,.05);}
.brand {display: flex; align-items: center; gap: .6rem;}
.brand-logo {width: 36px; height: 36px; border-radius: 10px; background: #4F46E5; color: #fff;
             display: flex; align-items: center; justify-content: center; font-weight: 700;}
.brand-name {font-weight: 700; font-size: 1.15rem; color: #0F172A; letter-spacing: -.01em;}
.user-chip {display: flex; align-items: center; gap: .5rem; justify-content: flex-end;}
.avatar {width: 34px; height: 34px; border-radius: 50%; background: #EEF2FF; color: #4338CA;
         display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: .85rem;}
.user-name {font-size: .88rem; color: #334155; font-weight: 500;}

/* page header */
.page-title {font-size: 1.65rem; font-weight: 700; color: #0F172A; margin: 0; letter-spacing: -.02em;}
.page-sub {color: #64748B; margin: .2rem 0 1.2rem 0; font-size: .95rem;}

/* cards and metrics */
[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
                          padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(15,23,42,.04);}
[data-testid="stMetricLabel"] {color: #64748B;}
[data-testid="stMetricValue"] {font-weight: 700; color: #0F172A;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
.face-card {padding: .9rem 1.1rem; border-radius: 12px; margin-bottom: .6rem;
            background: #EEF2FF; border: 1px solid #C7D2FE;}
.face-card h3 {margin: 0; font-size: 1.05rem; color: #312E81;}
.big-emoji {font-size: 2.4rem; line-height: 1;}

/* login */
.hero {background: #4F46E5; color: #fff; border-radius: 20px; padding: 2.6rem 2.2rem; min-height: 470px;}
.hero h1 {color: #fff; font-size: 2.3rem; font-weight: 700; margin: 0 0 .6rem 0; letter-spacing: -.02em;}
.hero p {color: #E0E7FF; font-size: 1.02rem; line-height: 1.6;}
.hero ul {list-style: none; padding: 0; margin: 1.6rem 0 0 0;}
.hero li {color: #fff; padding: .45rem 0; font-size: .97rem;}
.hero li span {display: inline-block; width: 26px;}
.hero .tag {display: inline-block; background: rgba(255,255,255,.15); padding: .25rem .7rem;
            border-radius: 999px; font-size: .78rem; margin-bottom: 1.2rem; color: #fff;}
.footer-note {text-align: center; color: #94A3B8; font-size: .8rem; margin-top: 2.5rem;}
</style>
""", unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<p class="page-title">{title}</p><p class="page-sub">{subtitle}</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading emotion model...")
def load_model():
    from model import EmotionModel
    return EmotionModel(str(WEIGHTS))


# ---------------------------------------------------------------- auth
def auth_page():
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("""<div class="hero">
            <span class="tag">Deep Learning · Computer Vision</span>
            <h1>MoodLens</h1>
            <p>Understand emotions through faces. A convolutional neural network trained on
            35,000+ images recognises seven emotions in photos and live video.</p>
            <ul>
              <li><span>📷</span> Analyse emotions in single or group photos</li>
              <li><span>🎥</span> Real-time webcam mood tracking</li>
              <li><span>📊</span> Personal dashboard with mood insights</li>
              <li><span>🔒</span> Secure accounts with hashed passwords</li>
            </ul></div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<p class="page-title" style="margin-top:1.5rem">Welcome back</p>'
                    '<p class="page-sub">Sign in to your account or create a new one.</p>',
                    unsafe_allow_html=True)
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            with st.form("login", border=False):
                u = st.text_input("Username", placeholder="vismaya")
                p = st.text_input("Password", type="password", placeholder="••••••••")
                if st.form_submit_button("Sign in", type="primary", width="stretch"):
                    user = db.verify(u.strip(), p)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    st.error("Wrong username or password.")
        with signup_tab:
            with st.form("signup", border=False):
                name = st.text_input("Full name", placeholder="Vismaya Katkar")
                u = st.text_input("Username", placeholder="3–20 letters, numbers or _")
                c1, c2 = st.columns(2)
                p1 = c1.text_input("Password", type="password", placeholder="Min 6 characters")
                p2 = c2.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    u = u.strip()
                    if not name.strip():
                        st.error("Enter your name.")
                    elif not re.fullmatch(r"[A-Za-z0-9_]{3,20}", u):
                        st.error("Username must be 3–20 letters, numbers or underscores.")
                    elif len(p1) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif p1 != p2:
                        st.error("Passwords don't match.")
                    elif not db.create_user(u, name.strip(), p1):
                        st.error("That username is taken. Try another.")
                    else:
                        st.success("Account created. Switch to Sign in to continue.")
    st.markdown('<p class="footer-note">MoodLens · Deep Learning Mini Project</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def get_df(user):
    df = pd.DataFrame(db.detections(user["username"]))
    if not df.empty:
        df["created"] = pd.to_datetime(df["created"])
    return df


def sessions_table(df):
    live = df[df["source"] == "live"]
    if live.empty:
        return pd.DataFrame()
    rows = []
    for sid, g in live.groupby("session_id"):
        start, end = g["created"].min(), g["created"].max()
        pos = positivity(g["emotion"])
        rows.append({"session": sid, "started": start,
                     "duration": f"{int((end - start).total_seconds()) // 60}m "
                                 f"{int((end - start).total_seconds()) % 60}s",
                     "dominant mood": label(g["emotion"].mode().iloc[0]),
                     "positivity": f"{pos:.0%}" if pos is not None else "-",
                     "readings": len(g)})
    return pd.DataFrame(rows).sort_values("started", ascending=False)


# ---------------------------------------------------------------- pages
def dashboard_page(user):
    page_header(f"Welcome back, {user['name'].split()[0]}", "Here's an overview of your mood insights.")
    df = get_df(user)
    c1, c2, c3, c4 = st.columns(4)
    if df.empty:
        for c, name in zip((c1, c2, c3, c4), ("Detections", "Live sessions", "Top mood", "Positivity")):
            c.metric(name, "-")
        st.info("No data yet. Open **Analyze** or start a **Live** session from the menu above.")
        return

    pos = positivity(df["emotion"])
    c1.metric("Detections", len(df))
    c2.metric("Live sessions", df.loc[df.source == "live", "session_id"].nunique())
    c3.metric("Top mood", label(df["emotion"].mode().iloc[0]))
    c4.metric("Positivity", f"{pos:.0%}" if pos is not None else "-",
              help="Happy + surprise as a share of all non-neutral moods")

    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Mood distribution")
        dist = df["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
        dist.index = [label(e) for e in dist.index]
        st.bar_chart(dist, color="#4F46E5")
    with right.container(border=True):
        st.subheader("Mood over time")
        daily = (df.assign(day=df["created"].dt.date)
                   .pivot_table(index="day", columns="emotion", values="id",
                                aggfunc="count", fill_value=0))
        st.line_chart(daily)

    box = st.container(border=True)
    box.subheader("Recent live sessions")
    s = sessions_table(df)
    if s.empty:
        box.caption("No live sessions yet.")
    else:
        s["started"] = s["started"].dt.strftime("%d %b %H:%M")
        box.dataframe(s.drop(columns="session").head(5), hide_index=True, width="stretch")


def analyze_page(user):
    page_header("Analyze photo", "Upload a photo or use your camera to detect emotions on every face.")
    if not WEIGHTS.exists():
        st.error("Model not found at models/emotion_cnn.pt. Train it first (see README).")
        return
    st.caption("Works best with front-facing, well-lit faces. Group photos are supported.")
    source = st.radio("Image source", ["Upload photo", "Use camera"], horizontal=True)
    file = (st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
            if source == "Upload photo" else st.camera_input("Take a photo"))
    if not file:
        return

    from model import draw_results
    data = file.getvalue()
    rgb = np.array(Image.open(file).convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    results = load_model().predict_faces(bgr)
    if not results:
        st.warning("No face detected. Try a clearer, front-facing photo.")
        st.image(rgb)
        return

    digest = hashlib.md5(data).hexdigest()[:12]
    saved = st.session_state.setdefault("saved", set())
    if digest not in saved:
        folder = UPLOADS / user["username"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now():%Y%m%d_%H%M%S}_{digest}.jpg"
        Image.fromarray(rgb).save(path, quality=90)
        for r in results:
            db.add_detection(user["username"], "photo", r["emotion"], r["confidence"],
                             image_path=str(path))
        saved.add(digest)

    left, right = st.columns([1.3, 1])
    with left:
        st.image(draw_results(bgr.copy(), results), channels="BGR",
                 caption=f"{len(results)} face(s) detected")
    with right:
        for n, r in enumerate(results, 1):
            st.markdown(f"""<div class="face-card"><span class="big-emoji">{EMOJI[r['emotion']]}</span>
                <h3>Face #{n}: {r['emotion'].capitalize()}</h3>
                Confidence: <b>{r['confidence']:.1%}</b></div>""", unsafe_allow_html=True)
            probs = pd.Series(r["probs"]).sort_values()
            probs.index = [label(e) for e in probs.index]
            st.bar_chart(probs, horizontal=True, height=220, color="#4F46E5")
            if r["confidence"] < 0.45:
                st.caption("Low confidence: the expression may be subtle or mixed.")


def live_page(user):
    page_header("Live session", "Track your emotions in real time through your webcam.")
    if not WEIGHTS.exists():
        st.error("Model not found at models/emotion_cnn.pt. Train it first (see README).")
        return
    st.write("Opens a webcam window that recognises your emotion in real time and logs your "
             "mood once per second to your dashboard. Press **Q** in that window to stop.")
    proc = st.session_state.get("live_proc")
    running = proc is not None and proc.poll() is None

    c1, c2 = st.columns(2)
    if c1.button("Start live session", icon=":material/videocam:", type="primary", disabled=running, width="stretch"):
        st.session_state.live_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "detect_live.py"), "--user", user["username"]], cwd=ROOT)
        st.rerun()
    if c2.button("Refresh results", icon=":material/refresh:", width="stretch"):
        st.rerun()
    if running:
        st.success("Session running. Look at the webcam window, and press Q there to finish.")

    df = get_df(user)
    s = sessions_table(df) if not df.empty else pd.DataFrame()
    if s.empty:
        st.info("No sessions recorded yet.")
        return
    last = s.iloc[0]
    st.subheader(f"Latest session: {last['started']:%d %b %Y, %H:%M}")
    g = df[df.session_id == last["session"]].sort_values("created")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", last["duration"])
    c2.metric("Dominant mood", last["dominant mood"])
    c3.metric("Positivity", last["positivity"])
    timeline = (g.assign(second=(g["created"] - g["created"].min()).dt.total_seconds(),
                         score=g["emotion"].map(lambda e: 1 if e in ("happy", "surprise")
                                                else 0 if e == "neutral" else -1))
                 .set_index("second")["score"])
    st.caption("Mood timeline (1 = positive, 0 = neutral, -1 = negative)")
    st.area_chart(timeline, color="#4F46E5")
    share = g["emotion"].value_counts(normalize=True).mul(100).round(1)
    share.index = [label(e) for e in share.index]
    st.bar_chart(share, color="#4F46E5")


def history_page(user):
    page_header("History", "Every detection you've made, from photos and live sessions.")
    df = get_df(user)
    if df.empty:
        st.info("No detections yet.")
        return
    c1, c2 = st.columns(2)
    src = c1.selectbox("Source", ["All", "photo", "live"])
    emo = c2.selectbox("Emotion", ["All"] + EMOTIONS)
    view = df if src == "All" else df[df.source == src]
    view = view if emo == "All" else view[view.emotion == emo]

    table = view[["id", "created", "source", "emotion", "confidence"]].copy()
    table["created"] = table["created"].dt.strftime("%d %b %Y, %H:%M:%S")
    table["emotion"] = table["emotion"].map(label)
    table["confidence"] = (table["confidence"] * 100).round(1)
    st.dataframe(table.rename(columns={"confidence": "confidence %"}), hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), "moodlens_history.csv", icon=":material/download:")

    photos = view[(view.source == "photo") & view.image_path.notna()].drop_duplicates("image_path")
    if not photos.empty:
        st.subheader("Analysed photos")
        cols = st.columns(4)
        for i, (_, r) in enumerate(photos.head(12).iterrows()):
            if Path(r["image_path"]).exists():
                cols[i % 4].image(r["image_path"], caption=label(r["emotion"]))

    with st.expander("Delete records"):
        rid = st.number_input("Record id", min_value=1, step=1)
        if st.button("Delete this record"):
            db.delete_detection(int(rid), user["username"])
            st.rerun()
        if st.button("Clear ALL my history"):
            db.clear(user["username"])
            st.rerun()


def about_page():
    page_header("About the model", "How MoodLens recognises emotions, and how well it performs.")
    st.write("MoodLens uses a **custom convolutional neural network built from scratch in "
             "PyTorch**, trained on the FER-2013 dataset (48×48 grayscale faces, 7 emotions). "
             "Faces are first located with OpenCV's Haar cascade detector, then each face is "
             "cropped, resized to 48×48 and classified by the CNN.")
    st.code("Input 1×48×48\n"
            "→ [Conv3×3-BN-ReLU ×2 → MaxPool → Dropout] ×4  (64→128→256→512 filters)\n"
            "→ Flatten (512×3×3) → Dense 256 → BN → ReLU → Dropout 0.5 → Dense 7 → Softmax",
            language=None)
    if METRICS.exists():
        m = json.loads(METRICS.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{m['test_accuracy']:.1%}")
        c2.metric("Parameters", f"{m['parameters'] / 1e6:.1f}M")
        c3.metric("Training images", f"{m['train_images']:,}")
        c4.metric("Epochs", m["epochs"])
        st.subheader("Accuracy per emotion")
        pc = pd.Series(m["per_class_accuracy"]).mul(100)
        pc.index = [label(e) for e in pc.index]
        st.bar_chart(pc, color="#4F46E5")
        st.caption("For reference, humans agree on FER-2013 labels only about 65% of the time, "
                   "so 65–70% is a strong result for this dataset.")
    else:
        st.info("Metrics appear here after training.")
    for name in ("training_curves", "confusion_matrix"):
        f = GRAPHS / f"{name}.png"
        if f.exists():
            st.subheader(name.replace("_", " ").title())
            st.image(str(f))


# ---------------------------------------------------------------- main
NAV = {"Dashboard": ":material/space_dashboard:", "Analyze": ":material/photo_camera:",
       "Live": ":material/videocam:", "History": ":material/history:", "About": ":material/info:"}


def navbar(user):
    with st.container(key="navbar"):
        brand, nav, account = st.columns([2.2, 5.6, 2.4], vertical_alignment="center")
        brand.markdown('<div class="brand"><div class="brand-logo">M</div>'
                       '<span class="brand-name">MoodLens</span></div>', unsafe_allow_html=True)
        with nav:
            choice = st.segmented_control("Navigation", list(NAV), key="nav",
                                          format_func=lambda p: f"{NAV[p]} {p}",
                                          label_visibility="collapsed")
        initials = "".join(w[0] for w in user["name"].split()[:2]).upper()
        a1, a2 = account.columns([1.6, 1], vertical_alignment="center")
        a1.markdown(f'<div class="user-chip"><div class="avatar">{initials}</div>'
                    f'<span class="user-name">{user["name"].split()[0]}</span></div>',
                    unsafe_allow_html=True)
        if a2.button("Logout", icon=":material/logout:", width="stretch"):
            st.session_state.clear()
            st.rerun()
    if choice is None:                      # clicking the active tab deselects it
        choice = st.session_state.get("last_page", "Dashboard")
    st.session_state.last_page = choice
    return choice


if "user" not in st.session_state:
    auth_page()
else:
    user = st.session_state.user
    st.session_state.setdefault("nav", "Dashboard")
    page = navbar(user)
    pages = {"Dashboard": dashboard_page, "Analyze": analyze_page,
             "Live": live_page, "History": history_page}
    pages[page](user) if page in pages else about_page()
    st.markdown('<p class="footer-note">MoodLens · Emotion recognition is approximate and '
                'not a medical or psychological tool.</p>', unsafe_allow_html=True)
