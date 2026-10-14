import streamlit as st
from transformers import pipeline

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="Sentiment Analyzer | Talha Siddiq", page_icon="🧠",
                   layout="centered", initial_sidebar_state="collapsed")

# ---------- STYLES ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #eef2ff 0%, transparent 60%),
                     radial-gradient(900px 500px at 110% 10%, #fdf2f8 0%, transparent 55%), #f8fafc; }
.block-container { padding-top: 2.2rem; max-width: 820px; }

.hero { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        border-radius: 18px; padding: 18px 22px; color: #fff;
        box-shadow: 0 20px 45px -18px rgba(79,70,229,.55); margin-bottom: 1.4rem; }
.hero .tag { display:inline-block; font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
             background: rgba(255,255,255,.18); padding:4px 10px; border-radius:999px; margin-bottom:12px; }
.stApp h1 { font-size:2.2rem !important; font-weight:800 !important; letter-spacing:-.03em; line-height:1.15;
             background: linear-gradient(135deg,#4f46e5,#7c3aed 55%,#db2777); -webkit-background-clip:text;
             background-clip:text; color:transparent !important; padding:.2rem 0 .4rem !important; }
.tag2 { display:inline-block; font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
        background:#eef2ff; color:#4338ca; padding:4px 10px; border-radius:999px; }
.hero p { color: rgba(255,255,255,.92); margin:0; font-size:.95rem; }
.chips { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.chip { font-size:.75rem; background: rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.25);
        padding:4px 10px; border-radius:999px; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background:#fff; padding:6px; border-radius:14px;
                                     border:1px solid #e5e7eb; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 16px; font-weight:600; color:#475569; }
.stTabs [aria-selected="true"] { background:#eef2ff; color:#4338ca !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

.stTextArea textarea, .stTextInput input { border-radius:12px !important; border:1px solid #e2e8f0 !important;
                                            background:#fff !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:#6366f1 !important;
                                                        box-shadow:0 0 0 3px rgba(99,102,241,.15) !important; }
.stButton > button { border-radius:12px; font-weight:600; padding:.6rem 1rem; border:0;
                     background: linear-gradient(135deg,#4f46e5,#7c3aed); color:#fff;
                     box-shadow:0 8px 20px -10px rgba(79,70,229,.7); transition: transform .12s ease; }
.stButton > button:hover { transform: translateY(-1px); color:#fff; }

.res { background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:16px 18px; margin:10px 0 4px;
       box-shadow:0 4px 14px -8px rgba(15,23,42,.12); }
.res .top { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.res .t { color:#64748b; font-size:.9rem; margin-bottom:10px; }
.pill { font-weight:700; font-size:.82rem; padding:5px 12px; border-radius:999px; }
.pill.pos { background:#dcfce7; color:#15803d; }
.pill.neg { background:#fee2e2; color:#b91c1c; }
.conf { font-weight:700; color:#0f172a; font-size:1.25rem; }
.bar { height:8px; background:#f1f5f9; border-radius:99px; overflow:hidden; margin-top:10px; }
.bar > div { height:100%; border-radius:99px; }
.low { font-size:.78rem; color:#b45309; margin-top:8px; }

.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:14px 0 6px; }
.stat { background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; }
.stat .k { font-size:.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em; }
.stat .v { font-size:1.5rem; font-weight:800; color:#0f172a; }

.foot { text-align:center; color:#94a3b8; font-size:.8rem; margin-top:2.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------- HERO (st.title includes Name + ID) ----------
st.markdown('<span class="tag2">CS4106 · Generative AI</span>', unsafe_allow_html=True)
st.title("Sentiment Analyzer — Talha Siddiq (B04-0923-000038)")

# ---------- MODEL ----------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return pipeline("sentiment-analysis", model=MODEL_ID)

clf = load_model()

def card(text, r, idx=None):
    pos = r["label"] == "POSITIVE"
    color = "#22c55e" if pos else "#ef4444"
    head = f"#{idx} · " if idx else ""
    low = '<div class="low">⚠️ Low confidence — this result may be unreliable.</div>' if r["score"] < 0.75 else ""
    st.markdown(f"""
    <div class="res">
      <div class="t">{head}“{text}”</div>
      <div class="top">
        <span class="pill {'pos' if pos else 'neg'}">{'😊 POSITIVE' if pos else '😞 NEGATIVE'}</span>
        <span class="conf">{r['score']:.1%}</span>
      </div>
      <div class="bar"><div style="width:{r['score']*100:.1f}%;background:{color}"></div></div>
      {low}
    </div>""", unsafe_allow_html=True)

# ---------- UI ----------
tab1, tab2 = st.tabs(["✍️  Single analysis", "📦  Batch mode (3 inputs)"])

with tab1:
    text = st.text_area("Your text", height=140, placeholder="e.g. The lecture today was really helpful!")
    if st.button("Analyze sentiment", use_container_width=True, key="single"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing…"):
                r = clf(text, truncation=True)[0]
            card(text, r)

with tab2:
    t1 = st.text_input("Text 1", placeholder="I love this course")
    t2 = st.text_input("Text 2", placeholder="The food was terrible")
    t3 = st.text_input("Text 3", placeholder="The lecture was okay")
    if st.button("Analyze all three", use_container_width=True, key="batch"):
        texts = [t for t in [t1, t2, t3] if t.strip()]
        if not texts:
            st.warning("Please enter at least one text.")
        else:
            with st.spinner("Analyzing batch…"):
                results = clf(texts, truncation=True)
            pos_n = sum(r["label"] == "POSITIVE" for r in results)
            st.markdown(f"""<div class="stats">
              <div class="stat"><div class="k">Analyzed</div><div class="v">{len(texts)}</div></div>
              <div class="stat"><div class="k">Positive</div><div class="v" style="color:#16a34a">{pos_n}</div></div>
              <div class="stat"><div class="k">Negative</div><div class="v" style="color:#dc2626">{len(texts)-pos_n}</div></div>
            </div>""", unsafe_allow_html=True)
            for i, (t, r) in enumerate(zip(texts, results), start=1):
                card(t, r, i)
