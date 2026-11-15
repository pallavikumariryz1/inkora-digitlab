from __future__ import annotations
import json
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

from modules.refinery import refine_digit
from modules.classifier import load_model, predict, extract_feature_maps, extract_first_layer_kernels, MODEL_PATH
from modules.agent import verify_sample

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
for folder in ["accepted", "rejected", "review_logs"]:
    (DATA / folder).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Digit Recognition Studio", page_icon="✎", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--paper:#eeece7;--paper2:#f8f7f3;--ink:#202b4b;--indigo:#27345f;--gold:#92743a;--goldline:#b59a62;--graphite:#62646b;}
.stApp{background:linear-gradient(135deg,#e8e7e3 0%,#f4f2ed 48%,#e3e2de 100%);color:var(--ink);font-family:'DM Sans',sans-serif}
.block-container{padding-top:1.25rem;max-width:1440px}
h1,h2,h3,h4,p,label,.stMarkdown{color:var(--ink)}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em}
h1{font-weight:700!important}
.stButton>button{border:1px solid var(--goldline);background:#29345b;color:#f8f7f3;border-radius:12px;font-weight:700;min-height:2.8rem}
.stButton>button:hover{border-color:#27345f;background:#394875;color:white}
div[data-testid="stSelectbox"]>div>div,div[data-testid="stNumberInput"] input{background:#fbfaf7;border:1px solid #b59a62;border-radius:10px;color:#202b4b}
div[data-testid="stAlert"]{border-radius:12px}
.hero{position:relative;overflow:hidden;border:1px solid #8d7950;border-radius:26px;padding:2rem 2.2rem;background:radial-gradient(circle at 50% 40%,#3c4774 0,#18223e 45%,#111827 100%);color:#f5f3ed;min-height:410px;display:flex;align-items:center;justify-content:space-between;gap:2rem}
.hero h1,.hero p{color:#f5f3ed}
.hero h1{font-size:clamp(2rem,4vw,3.5rem);line-height:1.08;max-width:610px}
.hero .gold{color:#d0b16c}
.hero-copy{position:relative;z-index:3;max-width:660px}
.globe{width:min(36vw,370px);height:min(36vw,370px);min-width:230px;min-height:230px;border-radius:50%;position:relative;flex-shrink:0;isolation:isolate;background:radial-gradient(circle at 31% 27%,#65718e 0,#35436c 29%,#202b4b 61%,#101625 100%);box-shadow:inset -30px -20px 48px #080d18,inset 10px 8px 28px #aab7df35,0 0 25px #7784bd55,0 0 58px #d0b16c22;border:1px solid #a7a9bd;overflow:hidden}
.globe:before{content:'';position:absolute;inset:0;border-radius:50%;background:radial-gradient(ellipse at 29% 24%,#ffffff38 0,transparent 35%),linear-gradient(105deg,transparent 36%,#c4d1f01c 50%,transparent 68%);z-index:3;pointer-events:none}
.globe:after{content:'';position:absolute;inset:2%;border-radius:50%;border:1px solid #d5c28b66;box-shadow:0 0 28px #b59a6244,inset 0 0 25px #0b1222;z-index:4;pointer-events:none}
.globe-grid{position:absolute;inset:0;border-radius:50%;overflow:hidden}
.globe-longitudes{position:absolute;inset:-3% 0;border-radius:50%;background:repeating-linear-gradient(90deg,transparent 0 22px,#aeb9d888 23px 24px,transparent 25px 45px);transform-origin:50% 50%;animation:globeTurn 13s linear infinite;filter:drop-shadow(0 0 3px #a8b9ed88);mask-image:radial-gradient(ellipse,black 48%,transparent 73%)}
.globe-latitudes{position:absolute;inset:0;border-radius:50%;background:repeating-linear-gradient(0deg,transparent 0 22px,#b7c1dc77 23px 24px,transparent 25px 45px);transform:perspective(380px) rotateX(5deg);mask-image:radial-gradient(ellipse,black 50%,transparent 74%)}
.globe-equator{position:absolute;left:3%;right:3%;top:49%;height:2px;background:linear-gradient(90deg,transparent,#e3c77d,#d8e3ff,#e3c77d,transparent);box-shadow:0 0 10px #d5b76f;opacity:.8}
@keyframes globeTurn{0%{transform:perspective(500px) rotateY(0deg) scaleX(1)}50%{transform:perspective(500px) rotateY(180deg) scaleX(.28)}100%{transform:perspective(500px) rotateY(360deg) scaleX(1)}}
.earth-orbit{position:absolute;inset:16% -8%;border:1px solid #d0b16c88;border-radius:50%;transform:rotate(-25deg);animation:orbit 11s linear infinite;z-index:5;pointer-events:none}
@keyframes orbit{to{transform:rotate(335deg)}}
.hero-kicker{color:#d4bc83!important;text-transform:uppercase;letter-spacing:.18em;font-size:.75rem;font-weight:800}
.pill{display:inline-block;border:1px solid #b59a62;color:#d8c38f;border-radius:99px;padding:.35rem .75rem;font-size:.78rem;font-weight:700}
.paper-card{background:#f9f8f4;border:1px solid #b5a274;border-radius:18px;padding:1.1rem 1.25rem;box-shadow:0 8px 25px #26314b0b}
.section-kicker{color:#80642e;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;font-weight:800}
.launch-stage{position:relative;overflow:hidden;border:1px solid #a48b56;border-radius:22px;padding:2rem;background:linear-gradient(180deg,#202a4a,#111827);color:white;text-align:center;min-height:260px}
.launch-stage h2,.launch-stage p{color:#f8f6ef}
.rocket{font-size:3.5rem;display:inline-block;animation:launch 2.4s ease-in-out infinite}
@keyframes launch{0%,100%{transform:translateY(14px) rotate(-45deg)}50%{transform:translateY(-22px) rotate(-45deg)}}
.splash{position:absolute;left:0;right:0;height:55%;pointer-events:none;opacity:0;filter:blur(1px)}
.splash.top{top:-60%;background:radial-gradient(ellipse at 50% 100%,#d8b45caa,transparent 65%),radial-gradient(ellipse at 20% 100%,#7783c9aa,transparent 55%)}
.splash.bottom{bottom:-60%;background:radial-gradient(ellipse at 50% 0%,#d8b45caa,transparent 65%),radial-gradient(ellipse at 80% 0%,#7783c9aa,transparent 55%)}
.splash.active{animation:splashIn 1.2s ease-out forwards}
@keyframes splashIn{0%{opacity:0;transform:scale(.6)}45%{opacity:1}100%{opacity:.0;transform:scale(1.5)}}
.neural-panel{background:#111318;border:1px solid #8e7b4e;border-radius:18px;padding:12px;overflow:hidden}
.neural-panel svg{width:100%;height:auto;display:block}
.neural-line{stroke-dasharray:5 7;animation:signal 2.2s linear infinite}
@keyframes signal{to{stroke-dashoffset:-48}}
.neuron{animation:pulse 2.4s ease-in-out infinite alternate;transform-box:fill-box;transform-origin:center}
@keyframes pulse{to{filter:drop-shadow(0 0 7px #e0bd6d);r:8}
}
.metric{font-family:'Space Grotesk';font-size:1.5rem;font-weight:700;color:#27345f}
.small-muted{color:#62646b;font-size:.9rem}
[data-testid="stCanvas"]{border:1px solid #b59a62;border-radius:12px;overflow:hidden}

/* Keep every button label readable across Streamlit themes and hover states. */
div.stButton>button,div.stButton>button p,div.stFormSubmitButton>button,div.stFormSubmitButton>button p,
button[kind="primary"],button[kind="primary"] p{color:#fff!important;font-weight:800!important;opacity:1!important}
div.stButton>button:hover,div.stButton>button:hover p,div.stFormSubmitButton>button:hover,
div.stFormSubmitButton>button:hover p{color:#fff!important;background:linear-gradient(110deg,#465786,#27345f)!important;box-shadow:0 0 0 2px #d0b16c66,0 8px 24px #27345f33;transform:translateY(-1px)}
div.stButton>button:focus,div.stFormSubmitButton>button:focus{color:#fff!important;outline:2px solid #c5a45e!important;outline-offset:2px}
button[kind="secondary"],button[kind="secondary"] p{color:#fff!important;background:#29345b!important}
.digit-shower{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.falling-digit{position:absolute;top:-15%;font-family:'Space Grotesk';font-weight:700;color:#d9bd75;text-shadow:0 0 12px #d9bd75aa;animation:digitFall 2.2s ease-in infinite}
.rising-digit{position:absolute;bottom:-15%;font-family:'Space Grotesk';font-weight:700;color:#9ca9d8;text-shadow:0 0 12px #9ca9d8aa;animation:digitRise 2.2s ease-out infinite}
@keyframes digitFall{to{transform:translateY(440px) rotate(190deg);opacity:0}}
@keyframes digitRise{to{transform:translateY(-440px) rotate(-190deg);opacity:0}}
.launch-stage{min-height:360px}
.launch-stage .rocket{animation:rocketLaunch 2.2s cubic-bezier(.2,.7,.3,1) infinite}
@keyframes rocketLaunch{0%{transform:translateY(65px) rotate(-45deg);filter:drop-shadow(0 0 0 #d5b76f)}45%{transform:translateY(0) rotate(-45deg);filter:drop-shadow(0 0 14px #d5b76f)}100%{transform:translateY(-95px) rotate(-45deg);filter:drop-shadow(0 0 24px #d5b76f);opacity:.25}}
.auth-card{background:#f9f8f4;border:1px solid #b59a62;border-radius:20px;padding:1.35rem;box-shadow:0 12px 35px #27345f12}
.auth-card h2{color:#202b4b}
.neural-caption{font-size:.82rem;color:#d7c58f;letter-spacing:.12em;font-weight:800;text-anchor:middle}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def neural_svg():
    """A directional teaching diagram: pixels -> learned features -> digit scores."""
    width, height = 980, 430
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Forward flow through a convolutional neural network">',
        '<defs><marker id="flowArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#d5b76f"/></marker></defs>'
    ]
    # Column headings
    headings = [(92, "INPUT PIXELS"), (330, "CONVOLUTION"), (565, "FEATURES"), (790, "CLASS SCORES")]
    for x, label in headings:
        parts.append(f'<text x="{x}" y="28" class="neural-caption">{label}</text>')

    # Input pixel patch
    parts.append('<rect x="43" y="78" width="98" height="112" rx="8" fill="#20232b" stroke="#d5b76f"/>')
    for r in range(7):
        for c in range(7):
            # Illustrative digit-like intensity field
            bright = (c in (2, 3, 4) and r in (0, 1, 5, 6)) or (r in (2, 3, 4) and c in (1, 5))
            fill = "#e8e4d9" if bright else "#454a56"
            parts.append(f'<rect x="{51+c*12}" y="{86+r*14}" width="10" height="12" rx="1" fill="{fill}"/>')
    parts.append('<text x="92" y="218" fill="#f4f0e6" font-size="13" text-anchor="middle">28 × 28 × 1</text>')
    parts.append('<text x="92" y="237" fill="#b9bdca" font-size="11" text-anchor="middle">784 values</text>')

    # Feature-map tiles
    for x, tint in [(290, "#42a8bd"), (326, "#d5b76f"), (362, "#d97965")]:
        parts.append(f'<rect x="{x}" y="95" width="58" height="74" rx="5" fill="#20232b" stroke="{tint}" stroke-width="2"/>')
        for r in range(4):
            for c in range(4):
                op = .25 + ((r*3+c*5) % 6)*.11
                parts.append(f'<rect x="{x+7+c*11}" y="{102+r*15}" width="8" height="11" fill="{tint}" opacity="{op:.2f}"/>')
    parts.append('<text x="330" y="218" fill="#f4f0e6" font-size="13" text-anchor="middle">Filters detect edges</text>')
    parts.append('<text x="330" y="237" fill="#b9bdca" font-size="11" text-anchor="middle">curves · corners · strokes</text>')

    # Feature hierarchy blocks
    for x, n, color in [(510, 5, "#42a8bd"), (548, 4, "#d5b76f"), (586, 3, "#d97965")]:
        for i in range(n):
            y = 83 + i*24
            parts.append(f'<rect x="{x}" y="{y}" width="27" height="17" rx="4" fill="{color}" opacity="{.38 + i*.1:.2f}" stroke="{color}"/>')
    parts.append('<text x="565" y="218" fill="#f4f0e6" font-size="13" text-anchor="middle">Pooling + deeper patterns</text>')
    parts.append('<text x="565" y="237" fill="#b9bdca" font-size="11" text-anchor="middle">compact feature maps</text>')

    # Dense hidden layer and ten output bars
    hidden_x = [710, 748]
    hidden_y = [88, 119, 150, 181, 212]
    for x in hidden_x:
        for y in hidden_y:
            parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#f4f0e6" stroke="#d5b76f" stroke-width="1.4"/>')
    output_x = 872
    for digit in range(10):
        y = 75 + digit*23
        parts.append(f'<text x="908" y="{y+4}" fill="#f4f0e6" font-size="12">{digit}</text>')
        parts.append(f'<rect x="868" y="{y-7}" width="25" height="13" rx="3" fill="#343b4c"/>')
        parts.append(f'<rect x="868" y="{y-7}" width="{5 + (digit*7)%20}" height="13" rx="3" fill="#d5b76f" opacity=".9"/>')
    parts.append('<text x="790" y="335" fill="#f4f0e6" font-size="13" text-anchor="middle">Dense layer combines evidence</text>')
    parts.append('<text x="885" y="335" fill="#f4f0e6" font-size="13" text-anchor="middle">Softmax</text>')

    # Directional arrows between stages
    for x1, x2 in [(151, 270), (450, 495), (620, 685), (770, 850)]:
        parts.append(f'<path d="M{x1} 275 L{x2} 275" stroke="#d5b76f" stroke-width="2" marker-end="url(#flowArrow)" opacity=".9"/>')
    parts.append('<text x="490" y="390" fill="#d5b76f" font-size="13" text-anchor="middle">FORWARD PASS: each layer transforms the signal before the next layer receives it →</text>')

    # Animated travelling signals along the four stage connectors.
    for x1, x2, color, delay in [(155, 265, "#42c4d9", "0s"), (455, 490, "#d5b76f", ".35s"), (625, 680, "#e17a68", ".7s"), (775, 845, "#f4f0e6", "1.05s")]:
        path_id = f"signal{int(x1)}"
        parts.append(f'<path id="{path_id}" d="M{x1} 275 L{x2} 275" fill="none" stroke="none"/>')
        parts.append(f'<circle r="5" fill="{color}" filter="drop-shadow(0 0 6px {color})"><animateMotion dur="1.7s" begin="{delay}" repeatCount="indefinite"><mpath href="#{path_id}"/></animateMotion></circle>')
    parts.append('</svg>')
    return "".join(parts)

def show_empty_dialog(slot_number):
    @st.dialog("A little blank space!")
    def _dialog():
        st.markdown("<div style='text-align:center;font-size:3.2rem;color:#a34b4b'>✕</div>", unsafe_allow_html=True)
        st.markdown(f"### Slot {slot_number} has no drawing yet")
        st.write("This studio learns from handwritten digits. Draw the digit you selected, then submit it so the model can analyse the strokes.")
        st.caption("Tip: use a single clear digit, leave a little margin, and avoid cutting the strokes at the edge.")
        if st.button("Back to my canvas", use_container_width=True):
            st.rerun()
    _dialog()

def save_record(folder, record):
    path = DATA / folder / f"sample_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

def landing():
    st.markdown("""
    <section class="hero">
      <div class="hero-copy">
        <div class="hero-kicker">DIGIT RECOGNITION STUDIO · LEARN BY EXPERIMENTING</div>
        <h1>The world is vast.<br><span class="gold">So are the opportunities.</span></h1>
        <p style="max-width:570px;color:#d6d9e2">A pencil stroke becomes pixels. Pixels become patterns. Patterns become a prediction. Take a chance and uncover how a neural network learns to recognise handwriting.</p>
        <span class="pill">DRAW · EXPLORE · UNDERSTAND DEEP LEARNING</span>
      </div>
      <div class="globe"><div class="globe-grid"><div class="globe-latitudes"></div><div class="globe-longitudes"></div><div class="globe-equator"></div></div><div class="earth-orbit"></div></div>
    </section>
    """, unsafe_allow_html=True)
    st.write("")
    c1,c2,c3 = st.columns([1,1,1])
    with c2:
        st.markdown("<div style='text-align:center;color:#27345f;font-weight:700'>Are you ready to see what your strokes can teach a model?</div>", unsafe_allow_html=True)
        if st.button("I'm taking the chance ✦", use_container_width=True):
            st.session_state.stage = "launch"
            st.rerun()

def launch():
    # The falling/rising digits visually meet around the rocket before the auth step.
    falling = "".join(
        f'<span class="falling-digit" style="left:{(i*13)%97}%;animation-delay:{(i%7)*-.28:.2f}s;font-size:{18+(i%4)*8}px">{i%10}</span>'
        for i in range(42)
    )
    rising = "".join(
        f'<span class="rising-digit" style="left:{(i*17+5)%97}%;animation-delay:{(i%8)*-.24:.2f}s;font-size:{16+(i%5)*7}px">{(i+5)%10}</span>'
        for i in range(42)
    )
    st.markdown(f"""
    <div class="launch-stage">
      <div class="digit-shower">{falling}{rising}</div>
      <div class="splash top active"></div><div class="splash bottom active"></div>
      <div class="rocket">🚀</div>
      <div class="hero-kicker">MISSION: RECOGNISE THE PATTERN</div>
      <h2>Every great model begins with one small signal.</h2>
      <p>Digits collide. Patterns emerge. Your learning journey is ready for launch.</p>
      <div style="font-size:1.8rem;letter-spacing:.35rem;color:#d5b76f">✦ 0 &nbsp; 1 &nbsp; 2 &nbsp; 3 &nbsp; 4 &nbsp; 5 &nbsp; 6 &nbsp; 7 &nbsp; 8 &nbsp; 9 ✦</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Continue to sign in ✦", use_container_width=True):
            st.session_state.stage = "auth"
            st.rerun()


def _auth_connection():
    connection = sqlite3.connect(ROOT / "data" / "users.sqlite")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    connection.commit()
    return connection


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)


def auth_page():
    st.markdown("""
    <section class="launch-stage" style="min-height:190px">
      <div class="hero-kicker">WELCOME TO YOUR LEARNING MISSION</div>
      <h2>One small sign-in. A universe of patterns.</h2>
      <p>Your drawings stay in this local project folder.</p>
      <div class="rocket" style="font-size:2.5rem">🚀</div>
    </section>
    """, unsafe_allow_html=True)
    st.write("")
    left, center, right = st.columns([1, 1.25, 1])
    with center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        mode = st.radio("Choose your entry", ["Sign in", "Create account"], horizontal=True, key="auth_mode")
        with st.form("local_auth_form", clear_on_submit=False):
            if mode == "Create account":
                name = st.text_input("Your name", placeholder="e.g. Alex")
                email = st.text_input("Email address", placeholder="you@example.com").strip().lower()
                password = st.text_input("Create password (8+ characters)", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create my account", use_container_width=True)
                if submitted:
                    if not name.strip() or not email or "@" not in email:
                        st.error("Please enter your name and a valid email address.")
                    elif len(password) < 8:
                        st.error("Please use a password with at least 8 characters.")
                    elif password != confirm:
                        st.error("Those passwords do not match. Please try again.")
                    else:
                        salt = secrets.token_bytes(16)
                        digest = _password_digest(password, salt)
                        conn = _auth_connection()
                        try:
                            conn.execute(
                                "INSERT INTO users(email,display_name,salt,password_hash,created_at) VALUES(?,?,?,?,?)",
                                (email, name.strip(), salt, digest, datetime.now(timezone.utc).isoformat())
                            )
                            conn.commit()
                            st.session_state.authenticated = True
                            st.session_state.auth_email = email
                            st.session_state.participant = name.strip()
                            st.session_state.canvas_version = st.session_state.get("canvas_version", 0) + 1
                            st.session_state.stage = "studio"
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("An account with this email already exists. Please sign in instead.")
                        finally:
                            conn.close()
            else:
                email = st.text_input("Email address", placeholder="you@example.com").strip().lower()
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in and enter studio", use_container_width=True)
                if submitted:
                    conn = _auth_connection()
                    row = conn.execute(
                        "SELECT display_name,salt,password_hash FROM users WHERE email=?", (email,)
                    ).fetchone()
                    conn.close()
                    if row and hmac.compare_digest(_password_digest(password, row[1]), row[2]):
                        st.session_state.authenticated = True
                        st.session_state.auth_email = email
                        st.session_state.participant = row[0]
                        st.session_state.canvas_version = st.session_state.get("canvas_version", 0) + 1
                        st.session_state.stage = "studio"
                        st.rerun()
                    else:
                        st.error("We couldn't verify those details. Check your email and password.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Local prototype authentication: passwords are salted and hashed, not stored as plain text. This is not a production identity service; use a unique test password, not your main email password.")
    if st.button("← Back to the launch screen"):
        st.session_state.stage = "launch"
        st.rerun()

def draw_slot(i, model):
    st.markdown(f'<div class="paper-card"><div class="section-kicker">DRAWING SLOT {i+1}</div><h3 style="margin:.15rem 0">Your handwritten digit</h3><span class="small-muted">Use the pencil to draw one digit on the paper.</span></div>', unsafe_allow_html=True)
    intended = st.selectbox("What digit did you intend to draw?", list(range(10)), key=f"intended_{i}")
    result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#26345f",
        background_color="#f8f7f2",
        height=220,
        width=260,
        drawing_mode="freedraw",
        update_streamlit=True,
        key=f"digit_canvas_{i}_{st.session_state.canvas_version}",
        return_image_data=True,
    )
    return intended, result.image_data

@st.cache_resource(show_spinner="Loading the trained digit-recognition model…")
def get_cached_model():
    return load_model()


def studio():
    st.markdown(f"## Welcome, {st.session_state.get('participant','learner')} ✎")
    st.markdown("Choose how many digits you want to experiment with. Each slot has its own drawing area and intended-label selector.")
    if st.button("Sign out", key="sign_out"):
        st.session_state.authenticated = False
        st.session_state.stage = "landing"
        st.session_state.canvas_version += 1
        st.rerun()
    count = st.select_slider("How many digits would you like to recognise?", options=list(range(1,11)), value=1, key="digit_count")
    st.caption(f"You selected {count} drawing slot{'s' if count != 1 else ''}.")
    st.session_state.canvas_version = st.session_state.get("canvas_version",0)
    if st.button("Clear all drawings", type="secondary"):
        st.session_state.canvas_version += 1
        st.rerun()

    model = None
    if MODEL_PATH.exists():
        try:
            model = get_cached_model()
            st.session_state.active_model = model
        except Exception as exc:
            st.error(f"The trained model could not be loaded: {exc}")
    else:
        st.info("The recognition model is not installed yet. Train it once with `python scripts/train_model.py`; the learning workspace will then run predictions.")

    slots = []
    cols_per_row = 2
    for start in range(0, count, cols_per_row):
        cols = st.columns(cols_per_row, gap="large")
        for offset, col in enumerate(cols):
            i = start + offset
            if i >= count: break
            with col:
                intended, rgba = draw_slot(i, model)
                slots.append((i, intended, rgba))

    st.write("")
    if st.button("Recognise my digit(s) →", type="primary", use_container_width=True):
        if model is None:
            st.warning("Train the CNN model first to enable digit recognition. Your drawings are not labelled as recognised without a real model.")
            return
        refined = []
        for i,intended,rgba in slots:
            image, diagnostics = refine_digit(rgba)
            if image is None:
                show_empty_dialog(i+1)
                return
            refined.append((i,intended,image,diagnostics))
        fingerprints = set()
        outcomes = []
        for i,intended,image,diagnostics in refined:
            predicted,confidence,probs = predict(model,image)
            outcome = verify_sample(image,diagnostics,intended,predicted,confidence,fingerprints)
            fingerprints.add(outcome["fingerprint"])
            record = {
                "participant": st.session_state.get("participant","Guest learner"),
                "slot": i+1, "intended": intended, "predicted": predicted,
                "confidence": confidence, "decision": outcome["decision"],
                "findings": outcome["findings"], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pixel_values": image.tolist()
            }
            save_record("accepted" if outcome["decision"]=="ACCEPTED" else "rejected", record)
            outcomes.append((record,image,probs,outcome))
        st.session_state.last_outcomes = outcomes
        st.session_state.stage = "results"
        st.rerun()

def activation_montage(feature_maps: np.ndarray, max_channels: int = 16) -> Image.Image:
    """Render real convolution activations as a compact grayscale montage."""
    maps = np.asarray(feature_maps)
    if maps.ndim != 3:
        raise ValueError(f"Expected HxWxC activations, received {maps.shape}.")
    channel_count = min(max_channels, maps.shape[-1])
    tile_size = 112
    columns = 4
    rows = int(np.ceil(channel_count / columns))
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + 22)), "#171923")
    draw = ImageDraw.Draw(canvas)
    for channel in range(channel_count):
        activation = maps[:, :, channel].astype(np.float32)
        low, high = float(activation.min()), float(activation.max())
        if high > low:
            normalized = (activation - low) / (high - low)
        else:
            normalized = np.zeros_like(activation)
        pixels = np.uint8(normalized * 255)
        tile = Image.fromarray(pixels, mode="L").resize((tile_size - 8, tile_size - 8), Image.Resampling.NEAREST).convert("RGB")
        x = (channel % columns) * tile_size + 4
        y = (channel // columns) * (tile_size + 22) + 3
        canvas.paste(tile, (x, y))
        draw.text((x + 3, y + tile_size - 5), f"Channel {channel + 1}", fill="#e2c477")
    return canvas


def kernel_montage(kernels: np.ndarray, max_channels: int = 16) -> Image.Image:
    """Display actual learned first-layer 3x3 kernels, not hand-drawn filters."""
    weights = np.asarray(kernels)
    if weights.ndim != 4:
        raise ValueError(f"Expected 3x3x1xN kernels, received {weights.shape}.")
    count = min(max_channels, weights.shape[-1])
    tile = 92
    columns = 8
    rows = int(np.ceil(count / columns))
    canvas = Image.new("RGB", (columns * tile, rows * (tile + 20)), "#171923")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        kernel = weights[:, :, 0, index].astype(np.float32)
        low, high = float(kernel.min()), float(kernel.max())
        scaled = (kernel - low) / (high - low) if high > low else np.zeros_like(kernel)
        pixels = np.uint8(scaled * 255)
        image = Image.fromarray(pixels, mode="L").resize((tile - 12, tile - 12), Image.Resampling.NEAREST).convert("RGB")
        x = (index % columns) * tile + 6
        y = (index // columns) * (tile + 20) + 3
        canvas.paste(image, (x, y))
        draw.text((x, y + tile - 4), f"Filter {index + 1}", fill="#e2c477")
    return canvas


def results():
    st.markdown("## Your model's response")
    outcomes = st.session_state.get("last_outcomes",[])
    if not outcomes:
        st.warning("No recognition results are available yet.")
        if st.button("Back to studio"): st.session_state.stage="studio"; st.rerun()
        return
    for idx,(record,image,probs,outcome) in enumerate(outcomes):
        st.markdown(f"### Slot {record['slot']} · Intended digit {record['intended']}")
        left,mid,right = st.columns([1,1.25,1.5])
        with left:
            enlarged = Image.fromarray(image).resize((224,224), Image.Resampling.NEAREST)
            st.image(enlarged, caption="Centred 28 × 28 input", width=224)
        with mid:
            st.metric("CNN prediction", str(record["predicted"]), f"{record['confidence']:.1%} confidence")
            st.markdown(f"**Agent decision: {record['decision']}**")
            if record["findings"]:
                for finding in record["findings"]:
                    st.warning(finding)
            else:
                st.success("The automated quality and label checks passed.")
        with right:
            chart = pd.DataFrame({"Probability": probs}, index=[str(i) for i in range(10)])
            st.bar_chart(chart, height=220)
    st.divider()
    st.markdown("### How the neural network understood your first digit")
    first_record,first_image,first_probs,_ = outcomes[0]
    st.markdown("### 1 · The exact image entering the CNN")
    st.caption("The refined drawing is normalised from 0–255 to 0–1 and reshaped to (1, 28, 28, 1).")
    st.code("Input tensor: (1, 28, 28, 1)\nConv2D(32, 3×3, ReLU, same): (1, 28, 28, 32)\nMaxPool(2×2): (1, 14, 14, 32)\nConv2D(64, 3×3, ReLU, same): (1, 14, 14, 64)\nMaxPool(2×2): (1, 7, 7, 64)\nFlatten: 3,136 values → Dense(128) → Dropout(0.25) → Dense(10, Softmax)", language="text")
    try:
        actual_maps = extract_feature_maps(st.session_state.get("active_model"), first_image)
        first_kernels = extract_first_layer_kernels(st.session_state.get("active_model"))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 2 · What did the first convolution filters learn?")
            st.image(kernel_montage(first_kernels), caption="Actual learned 3×3 kernels from feature_maps_1", use_container_width=True)
            st.caption("These are learned numerical weights. A filter is not assigned a guaranteed human label such as 'vertical edge'; its response depends on the input.")
        with c2:
            st.markdown("### 3 · What did those filters respond to?")
            st.image(activation_montage(actual_maps["feature_maps_1"]), caption="Real activation maps · Conv2D layer 1 · first 16 of 32 channels", use_container_width=True)
        st.markdown("### 4 · Deeper patterns from the second convolution")
        st.image(activation_montage(actual_maps["feature_maps_2"]), caption="Real activation maps · Conv2D layer 2 · first 16 of 64 channels", use_container_width=True)
        st.caption("Bright regions indicate stronger activation in that channel for this particular drawing. Each channel is independently scaled for visibility.")
    except Exception as exc:
        st.warning(f"The prediction is available, but live feature maps could not be extracted: {exc}")
        st.info("The feature-map extractor explicitly builds the model before reading its intermediate layer outputs. If this message persists, retrain and resave the model using the current scripts/train_model.py.")
    st.markdown("### 5 · How did the model choose its answer?")
    st.latex(r"P(y=i)=\frac{e^{z_i}}{\sum_{j=0}^{9}e^{z_j}}")
    st.write(f"For this drawing, the model selected **{first_record['predicted']}** with a probability of **{first_record['confidence']:.2%}**. The bar chart above shows all ten actual class probabilities.")
    st.markdown("### 6 · Forward propagation is not backpropagation")
    forward_col, backward_col = st.columns(2)
    with forward_col:
        st.markdown("**Inference · when you click Recognise**")
        st.code("image → convolution → pooling →\nfeatures → dense layers → softmax", language="text")
        st.caption("The trained weights are used as they are. This drawing does not update the model.")
    with backward_col:
        st.markdown("**Training · when the CNN learns from MNIST**")
        st.code("label + prediction → loss → gradients\n→ backpropagation → Adam weight update", language="text")
        st.latex(r"L=-\log(p_{\mathrm{correct}})")
        st.latex(r"W_{new}=W_{old}-\eta\frac{\partial L}{\partial W}")
        st.caption("This is the gradient-descent intuition. Adam uses adaptive first- and second-moment estimates to update parameters.")
    if st.button("Try another drawing", use_container_width=True):
        st.session_state.canvas_version += 1
        st.session_state.stage = "studio"
        st.rerun()

if "stage" not in st.session_state: st.session_state.stage = "landing"
if "canvas_version" not in st.session_state: st.session_state.canvas_version = 0

if st.session_state.stage == "landing":
    landing()
elif st.session_state.stage == "launch":
    launch()
elif st.session_state.stage == "auth":
    auth_page()
elif st.session_state.stage == "studio":
    studio()
elif st.session_state.stage == "results":
    results()
