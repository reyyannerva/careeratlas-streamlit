# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "app" / "jobs_app_with_link_health.parquet"


import io
import json
import re
import html as _html
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ===================== UI CORE =====================

COMPASS_SVG = r"""
<svg width="44" height="44" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="8" y1="8" x2="58" y2="58">
      <stop stop-color="#6EE7FF"/><stop offset="1" stop-color="#B46CFF"/>
    </linearGradient>
    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2.2" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <circle cx="32" cy="32" r="26" stroke="url(#g)" stroke-width="3" filter="url(#glow)"/>
  <circle cx="32" cy="32" r="3.4" fill="url(#g)"/>
  <path d="M38 26 L50 14 L39 39 L14 50 L26 26 Z" stroke="url(#g)" stroke-width="2"
        fill="rgba(110,231,255,0.10)"/>
</svg>
"""

def ca_html(s: str, **_ignored):
    """
    Render raw HTML safely in Streamlit.
    - Removes indentation that causes markdown code-block rendering
    - Always uses unsafe_allow_html=True
    - Accepts extra kwargs (ignored) so accidental unsafe_allow_html arg won't crash
    """
    s2 = textwrap.dedent(str(s)).strip("\n")
    st.markdown(s2, unsafe_allow_html=True)


def inject_css():
    
    st.markdown("<style>section[data-testid='stSidebar'] pre, section[data-testid='stSidebar'] code{display:none!important;}</style>", unsafe_allow_html=True)
    st.markdown(r"""
<style>
:root{
  --bg0:#050712; --bg1:#070B18;
  --text:#EAF1FF;
  --muted:rgba(234,241,255,.72);
  --muted2:rgba(234,241,255,.55);
  --line:rgba(233,238,255,.12);
  --line2:rgba(233,238,255,.18);
  --cyan:#6EE7FF;
  --violet:#B46CFF;
  --good:#2EE59D;
  --warn:#FFCC66;
  --shadow:0 16px 46px rgba(0,0,0,.50);
  --r:18px;
}

html, body, [data-testid="stAppViewContainer"]{ color: var(--text) !important; }
.stApp{
  background:
    radial-gradient(1200px 900px at 12% 10%, rgba(110,231,255,0.16), transparent 60%),
    radial-gradient(1200px 900px at 88% 18%, rgba(180,108,255,0.16), transparent 60%),
    radial-gradient(900px 700px at 50% 95%, rgba(255,204,102,0.10), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}
.block-container{ max-width: 1180px; padding-top: 1.05rem; padding-bottom: 2.2rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(8,12,28,.92), rgba(6,9,20,.92));
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] *{ color: var(--text) !important; }
.stCaption, [data-testid="stCaptionContainer"]{ color: var(--muted) !important; }

.ca-hero{
  border:1px solid rgba(233,238,255,.14);
  border-radius: 22px;
  padding: 18px 20px;
  background:
    radial-gradient(900px 360px at 10% 0%, rgba(110,231,255,.20), transparent 60%),
    radial-gradient(900px 360px at 90% 10%, rgba(180,108,255,.18), transparent 60%),
    linear-gradient(180deg, rgba(16,24,52,.86), rgba(10,16,34,.74));
  box-shadow: var(--shadow);
  backdrop-filter: blur(6px);
}
.ca-title{
  font-size: 44px;
  font-weight: 950;
  letter-spacing: -0.04em;
  margin: 0;
  line-height: 1.02;
  background: linear-gradient(90deg, #FFFFFF, rgba(234,241,255,.72));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.ca-sub{ margin-top: 8px; color: var(--muted); font-size: 14px; line-height: 1.5; }
.ca-kpi{ margin-top: 10px; color: var(--muted2); font-size: 12px; }

.ca-card{
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 16px 18px;
  background: linear-gradient(180deg, rgba(15,24,54,.82), rgba(10,16,34,.76));
  box-shadow: var(--shadow);
  backdrop-filter: blur(4px);
}

.ca-badge{
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  border:1px solid var(--line);
  background: rgba(255,255,255,0.05);
  margin-right:8px; margin-bottom:8px;
  font-size: .88rem;
  color: var(--text);
}
.ca-badge-skill{ border-color: rgba(110,231,255,0.45); }
.ca-badge-miss { border-color: rgba(255,204,102,0.45); }
.ca-badge-good { border-color: rgba(46,229,157,0.45); }

input, textarea{
  background: rgba(10,14,26,.85) !important;
  border: 1px solid rgba(234,241,255,.18) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
  caret-color: var(--cyan) !important;
}
input::placeholder, textarea::placeholder{ color: rgba(234,241,255,.45) !important; }

div[data-baseweb="select"] > div{
  background: rgba(10,14,26,.85) !important;
  color: var(--text) !important;
  border: 1px solid rgba(234,241,255,.18) !important;
  border-radius: 14px !important;
}
div[data-baseweb="menu"]{
  background: rgba(9,12,24,.98) !important;
  border: 1px solid rgba(234,241,255,.18) !important;
  border-radius: 14px !important;
}
div[data-baseweb="menu"] *{ color: var(--text) !important; }

[data-testid="stVerticalBlockBorderWrapper"]{
  background: rgba(10,16,34,.52) !important;
  border: 1px solid rgba(234,241,255,.10) !important;
  border-radius: 18px !important;
  box-shadow: 0 12px 34px rgba(0,0,0,.28) !important;
}

div[data-testid="stMetric"]{
  background: rgba(10,16,34,.55) !important;
  border: 1px solid rgba(234,241,255,.14) !important;
  border-radius: 16px !important;
  padding: 12px 12px !important;
}
div[data-testid="stMetric"] *{ color: var(--text) !important; }

div.stButton > button{
  border-radius: 14px !important;
  border: 1px solid rgba(110,231,255,0.45) !important;
  background: linear-gradient(135deg, rgba(110,231,255,0.18), rgba(180,108,255,0.18)) !important;
  color: var(--text) !important;
  padding: 0.60rem 1rem !important;
  font-weight: 850 !important;
  box-shadow: 0 10px 26px rgba(0,0,0,0.28) !important;
  transition: transform .12s ease, filter .12s ease !important;
}
div.stButton > button:hover{
  border-color: rgba(110,231,255,0.75) !important;
  transform: translateY(-1px) !important;
  filter: brightness(1.08) !important;
}

div[data-testid="stLinkButton"] a{
  display:inline-flex !important; align-items:center !important; justify-content:center !important;
  width:100% !important; text-decoration:none !important;
  border-radius: 14px !important;
  border: 1px solid rgba(110,231,255,0.45) !important;
  background: linear-gradient(135deg, rgba(110,231,255,0.18), rgba(180,108,255,0.18)) !important;
  color: var(--text) !important;
  padding: 0.60rem 1rem !important;
  font-weight: 850 !important;
  box-shadow: 0 10px 26px rgba(0,0,0,0.28) !important;
}
div[data-testid="stLinkButton"] a:hover{
  border-color: rgba(110,231,255,0.75) !important;
  transform: translateY(-1px) !important;
  filter: brightness(1.08) !important;
}

pre, code, [data-testid="stCodeBlock"], [data-testid="stJson"]{
  background: rgba(10,14,26,.92) !important;
  color: var(--text) !important;
  border: 1px solid rgba(234,241,255,.16) !important;
  border-radius: 16px !important;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def badge_row(items, cls="ca-badge ca-badge-skill", maxn=18):
    items = [str(x) for x in (items or []) if str(x).strip()]
    if not items:
        return
    html = " ".join([f"<span class='{cls}'>{_html.escape(str(x))}</span>" for x in items[:maxn]])
    ca_html(html)

def render_table_dark(df: pd.DataFrame, height_px: int = 360):
    if df is None or df.empty:
        st.caption("Tablo boş.")
        return
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    rows_html = []
    for _, row in d.iterrows():
        tds = "".join([f"<td>{_html.escape(str(row[c]))}</td>" for c in d.columns])
        rows_html.append(f"<tr>{tds}</tr>")
    thead = "".join([f"<th>{_html.escape(str(c))}</th>" for c in d.columns])

    ca_html(f"""
<div class="ca-card" style="padding:12px 12px;">
  <div style="max-height:{height_px}px; overflow:auto; border-radius:14px; border:1px solid rgba(233,238,255,.12);">
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
      <thead>
        <tr style="background: rgba(12,18,36,.95); position:sticky; top:0;">
          {thead}
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>
</div>
<style>
table th, table td {{
  padding: 10px 10px;
  border-bottom: 1px solid rgba(233,238,255,.10);
  color: rgba(234,241,255,.92);
  text-align: left;
  white-space: nowrap;
}}
table tbody tr:hover {{ background: rgba(110,231,255,.06); }}
</style>
""")

# ===================== DATASET =====================

# ===================== DATASET =====================

def pick_dataset() -> Path | None:
    p = DATASET_PATH
    return p if p.exists() else None


@st.cache_data(show_spinner=False)
def load_df(path_str: str) -> pd.DataFrame:
    df = pd.read_parquet(path_str)
    need_cols = ["title","company","location","apply_url","url","skills_str","source","description","link_ok","link_reason"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = "" if c != "link_ok" else True
    for c in ["title","company","location","apply_url","url","skills_str","source","description","link_reason"]:
        df[c] = df[c].astype(str).fillna("")
    if "link_ok" in df.columns:
        df["link_ok"] = df["link_ok"].fillna(True).astype(bool)
    return df

def clean_desc_html(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _html.unescape(s)
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*p\s*>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s).strip()
    return s

# ===================== FAVORITES =====================

def fav_key(r: dict) -> str:
    return f"{r.get('apply_url') or r.get('url') or ''}||{r.get('title') or ''}||{r.get('company') or ''}"

def fav_store() -> dict:
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = {}
    return st.session_state["favorites"]

def fav_add(r: dict):
    fav = fav_store()
    k = fav_key(r)
    fav[k] = {
        "title": r.get("title","") or "",
        "company": r.get("company","") or "",
        "location": r.get("location","") or "",
        "source": r.get("source","") or "",
        "url": r.get("apply_url","") or r.get("url","") or "",
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state["favorites"] = fav

def fav_remove(k: str):
    fav = fav_store()
    if k in fav:
        fav.pop(k)
    st.session_state["favorites"] = fav

# ===================== API (Optional) =====================

def api_base() -> str:
    return st.session_state.get("api_base", "http://127.0.0.1:8000")

def api_get(path: str, timeout=20):
    url = api_base().rstrip("/") + path
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict, timeout=60):
    url = api_base().rstrip("/") + path
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def render_kg_pyvis(nodes, edges, height=560):
    try:
        from pyvis.network import Network
    except Exception:
        st.info("pyvis yok. KG için JSON indir kullan.")
        return
    net = Network(height=f"{height}px", width="100%", bgcolor="#0B1020", font_color="#EAF1FF", directed=False)
    net.barnes_hut(gravity=-25000, central_gravity=0.2, spring_length=160, spring_strength=0.03, damping=0.8)
    for n in nodes or []:
        nid = n.get("id") or n.get("key") or n.get("name")
        label = n.get("label") or n.get("name") or str(nid)
        ntype = n.get("type") or n.get("node_type") or "node"
        size = 24 if ntype == "role" else (16 if ntype == "skill" else 13)
        color = "#6EE7FF" if ntype == "role" else ("#B46CFF" if ntype == "course" else "#2EE59D")
        net.add_node(str(nid), label=label, title=f"{label} ({ntype})", color=color, size=size)
    for e in edges or []:
        s = e.get("source") or e.get("from")
        t = e.get("target") or e.get("to")
        if s is None or t is None:
            continue
        net.add_edge(str(s), str(t), title=str(e.get("type") or e.get("relation") or ""))
    components.html(net.generate_html(), height=height, scrolling=True)

# ===================== SIMPLE ROLE/SKILL UTIL =====================

SKILL_VOCAB = [
 "python","pandas","numpy","scikit-learn","sklearn","tensorflow","pytorch","keras","xgboost","lightgbm",
 "statistics","ml","machine learning","deep learning","nlp","llm","transformers",
 "sql","postgres","mysql","bigquery","snowflake","dbt","airflow","spark","databricks",
 "tableau","powerbi","excel","analytics","dashboard","etl","data warehouse",
 "java","go","golang","node","nodejs","fastapi","flask","django","spring","api","microservices",
 "docker","kubernetes","k8s","terraform","aws","gcp","azure","linux","git","ci/cd","cicd",
 "prometheus","grafana","sre","devops","redis","kafka",
 "security","siem","soc","pentest","vulnerability","threat","incident response",
 "product","roadmap","agile","scrum","stakeholder","metrics","experiment"
]

ROLE_PRESETS = {
 "Veri Bilimci": "python machine learning statistics pytorch tensorflow pandas scikit-learn model evaluation experimentation",
 "Veri Analisti": "sql dashboards analytics reporting bi tableau powerbi metrics stakeholder excel",
 "Veri Mühendisi": "sql etl data warehouse airflow spark dbt docker linux aws azure gcp kafka",
 "Backend Engineer": "python java go nodejs backend api microservices postgres redis kafka docker kubernetes",
 "MLOps / ML Engineer": "mlops deployment pipelines feature store monitoring airflow docker kubernetes",
 "SRE / DevOps": "sre devops reliability monitoring prometheus grafana terraform aws gcp incident kubernetes",
 "Product Manager": "product management roadmap discovery metrics experiments stakeholders agile",
 "Cyber Security": "security pentest soc siem threat detection incident response vulnerability assessment",
}

def normalize_text(s: str) -> str:
    s = (s or "")
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_skills_from_text(text: str, topk: int = 30):
    tx = normalize_text(text)
    hits = []
    for sk in SKILL_VOCAB:
        if sk.lower() in tx:
            hits.append("scikit-learn" if sk == "sklearn" else sk)
    out = []
    seen = set()
    for h in hits:
        hh = h.lower()
        if hh not in seen:
            seen.add(hh)
            out.append(h)
    return out[:topk]

def infer_role_from_text(text: str):
    tx = normalize_text(text)
    scores = {}
    for role, q in ROLE_PRESETS.items():
        toks = [t.strip().lower() for t in q.split() if t.strip()]
        scores[role] = sum(1 for t in toks if t in tx)
    best = max(scores, key=scores.get) if scores else "Veri Bilimci"
    return best, scores

def parse_skills_csv(s: str):
    if not s:
        return []
    return [x.strip().lower() for x in str(s).split(",") if x.strip()]

def row_to_dict(r) -> dict:
    return {
        "title": (r.get("title","") or ""),
        "company": (r.get("company","") or ""),
        "location": (r.get("location","") or ""),
        "source": (r.get("source","") or ""),
        "apply_url": (r.get("apply_url","") or ""),
        "url": (r.get("url","") or ""),
        "skills_str": (r.get("skills_str","") or ""),
        "description": (r.get("description","") or ""),
        "link_ok": bool(r.get("link_ok", True)),
        "link_reason": (r.get("link_reason","") or ""),
    }

# ===================== PAGES =====================

def page_home(df: pd.DataFrame):
    """
    HOME: promo/hero
    IMPORTANT: HTML must NOT be rendered as code-block.
    Fix: dedent + lstrip before st.markdown.
    (Only Home is changed. Other pages untouched.)
    """
    dataset = _html.escape(st.session_state.get("dataset_path", "-"))

    st.markdown(r"""
<style>
/* HOME-only safety: if any pre/code appears in Home, hide it (doesn't affect other pages) */
.ca-home pre, .ca-home code { display:none !important; }

/* HERO */
.ca-home .hero{
  position:relative; overflow:hidden;
  border-radius:24px;
  padding:28px 26px;
  border:1px solid rgba(233,238,255,.16);
  background: linear-gradient(135deg, rgba(7,10,18,.82), rgba(11,18,32,.72));
  box-shadow: 0 28px 90px rgba(0,0,0,.58);
}
.ca-home .hero:before{
  content:"";
  position:absolute; inset:-2px;
  background:
    radial-gradient(900px 520px at 15% 10%, rgba(110,231,255,.42), transparent 62%),
    radial-gradient(900px 620px at 85% 12%, rgba(180,108,255,.46), transparent 64%),
    radial-gradient(1000px 520px at 45% 120%, rgba(255,204,102,.22), transparent 66%);
  pointer-events:none;
}
.ca-home .hero:after{
  content:"";
  position:absolute; inset:0;
  background: rgba(4,6,14,.56);
  pointer-events:none;
}
.ca-home .hero > *{ position:relative; z-index:2; }

.ca-home .logoBox{
  width:86px; height:86px; border-radius:26px;
  border:1px solid rgba(233,238,255,.20);
  background:linear-gradient(135deg, rgba(110,231,255,.16), rgba(180,108,255,.16));
  box-shadow:0 18px 58px rgba(0,0,0,.60);
  display:flex; align-items:center; justify-content:center;
}
.ca-home .title{
  font-size:64px;
  font-weight:950;
  letter-spacing:-0.07em;
  line-height:1.02;
  color:#EAF1FF !important;
  text-shadow: 0 14px 46px rgba(0,0,0,.70), 0 2px 0 rgba(0,0,0,.35);
}
.ca-home .sub{
  font-size:15.5px;
  line-height:1.55;
  max-width:980px;
  margin-top:6px;
  color: rgba(234,241,255,.92) !important;
  text-shadow: 0 10px 26px rgba(0,0,0,.55);
}
.ca-home .chip{
  display:inline-flex; align-items:center; gap:8px;
  padding:7px 11px;
  border-radius:999px;
  border:1px solid rgba(233,238,255,.18);
  background: rgba(10,16,34,.58);
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
  font-weight:850; font-size:12.5px;
  color:#EAF1FF !important;
}
.ca-home .glass{
  margin-top:16px;
  border-radius:18px;
  padding:14px 14px;
  border:1px solid rgba(233,238,255,.14);
  background: linear-gradient(135deg, rgba(10,16,34,.82), rgba(14,22,48,.68));
}
.ca-home .h{
  font-weight:950;
  font-size:14px;
  color:#EAF1FF !important;
}
.ca-home .p{
  margin-top:6px;
  line-height:1.70;
  color: rgba(234,241,255,.90) !important;
}
.ca-home .muted{
  color: rgba(234,241,255,.62) !important;
  font-size:12px;
  word-break:break-all;
}

/* Steps card */
.ca-home .steps{
  margin-top:14px;
  border-radius:18px;
  padding:14px 14px;
  border:1px solid rgba(233,238,255,.14);
  background: linear-gradient(135deg, rgba(10,16,34,.78), rgba(14,22,48,.62));
}
.ca-home .steps .p{ color: rgba(234,241,255,.86) !important; line-height:1.70; }
</style>
""", unsafe_allow_html=True)

    hero_html = f"""
<div class="ca-home">
  <div class="hero">
    <div style="display:flex; align-items:center; gap:16px;">
      <div class="logoBox">
        <div style="transform:scale(1.55); line-height:0;">{COMPASS_SVG}</div>
      </div>
      <div style="flex:1;">
        <div class="title">CareerAtlas</div>
        <div class="sub">
          CV’nden otomatik <b>rol &amp; skill</b> çıkarır, eksikleri önceliklendirir.
          İlanları akıllıca sıralar. <b>Learning path</b> + <b>Knowledge Graph</b> ile kariyerini uzay temalı bir haritaya çevirir.
        </div>
        <div style="margin-top:12px; display:flex; flex-wrap:wrap; gap:10px;">
          <span class="chip">🧭 <b>CV → Role/Skill</b></span>
          <span class="chip">🧠 <b>Semantic Match</b></span>
          <span class="chip">🕸️ <b>Knowledge Graph</b></span>
          <span class="chip">🎤 <b>Interview Coach</b></span>
        </div>
      </div>
    </div>

    <div class="glass">
      <div style="display:flex; gap:16px; flex-wrap:wrap;">
        <div style="flex:1; min-width:300px;">
          <div class="h">🚀 3 adımda demo</div>
          <div class="p">
            1) <b>CV → Career Advisor</b> (PDF)<br/>
            2) <b>Eksik skill</b> + <b>learning path</b> + <b>KG</b><br/>
            3) <b>Model (semantic)</b> → ⭐ Favoriler → 🧬 Dijital ikiz
          </div>
        </div>
        <div style="min-width:320px;">
          <div class="h">📦 Dataset</div>
          <div class="p">
            <span>Kayıt: <b>{len(df)}</b></span><br/>
            <span class="muted">{dataset}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""
    st.markdown(textwrap.dedent(hero_html).lstrip("\n").rstrip(), unsafe_allow_html=True)

    # KPI (dokunmuyoruz)
    c1, c2, c3 = st.columns(3)
    c1.metric("İlan sayısı", f"{len(df)}")
    c2.metric("Kaynak", f"{df['source'].nunique() if 'source' in df.columns else '-'}")
    c3.metric("Link OK", f"{df['link_ok'].mean():.2%}" if "link_ok" in df.columns else "-")

    steps_html = """
<div class="ca-home">
  <div class="steps">
    <div class="h">📌 Nereden başlamalı?</div>
    <div class="p">
      <b>1)</b> CV → Career Advisor • PDF yükle • <b>Kariyer Haritamı Çıkar</b><br/>
      <b>2)</b> Model (semantic) • ilanları sırala • ⭐ Favori’ye ekle<br/>
      <b>3)</b> Dijital İkiz • profil + favoriler ile tek sayfada özet
    </div>
  </div>
</div>
"""
    st.markdown(textwrap.dedent(steps_html).lstrip("\n").rstrip(), unsafe_allow_html=True)








def page_listings(df: pd.DataFrame):
    st.subheader("🗂️ İlanlar")
    with st.sidebar:
        st.caption("Filtreler")
        q = st.text_input("Başlık/Şirket ara", "")
        srcs = sorted([s for s in df["source"].unique().tolist() if s])
        pick_src = st.multiselect("Kaynak", srcs, default=srcs)
        skill = st.text_input("Skill (örn: python)", "")
        only_ok = st.toggle("Sadece link sağlam", value=False)
        page_size = st.slider("Sayfa boyutu", 10, 100, 30, 10)

    d = df.copy()
    if pick_src:
        d = d[d["source"].isin(pick_src)]
    if q.strip():
        qq = q.strip().lower()
        d = d[d["title"].str.lower().str.contains(qq, na=False) | d["company"].str.lower().str.contains(qq, na=False)]
    if skill.strip():
        sk = skill.strip().lower()
        d = d[d["skills_str"].str.lower().str.contains(sk, na=False)]
    if only_ok and "link_ok" in d.columns:
        d = d[d["link_ok"].astype(bool)]

    st.caption(f"Sonuç: **{len(d)}**")
    total = len(d)
    pages = max(1, (total + page_size - 1) // page_size)
    pno = st.number_input("Sayfa", min_value=1, max_value=pages, value=1, step=1)
    start = (pno - 1) * page_size
    end = min(total, start + page_size)
    d = d.iloc[start:end].copy()

    for i, r in d.iterrows():
        rr = row_to_dict(r)
        title = (rr["title"] or "").strip() or "(Başlık yok)"
        company = (rr["company"] or "").strip()
        loc = (rr["location"] or "").strip()
        url = (rr["apply_url"] or rr["url"] or "").strip()
        ok = bool(rr.get("link_ok", True))
        reason = str(rr.get("link_reason","") or "")
        badge = "🟢 Link Sağlam" if ok else f"🔴 Link Kırık ({reason})"

        with st.container(border=True):
            top = st.columns([5,1.5,1.5])
            top[0].markdown(f"### {title}")
            top[0].caption(f"**{company}** • {loc if loc else 'Lokasyon yok'} • {badge}")

            if url:
                try:
                    top[1].link_button("Başvur", url, use_container_width=True)
                except Exception:
                    top[1].markdown(f"[Başvur]({url})")

            k = fav_key(rr)
            if k in fav_store():
                if top[2].button("⭐ Kaldır", key=f"fav_rm_list_{i}", use_container_width=True):
                    fav_remove(k); st.toast("Favoriden kaldırıldı", icon="⭐")
            else:
                if top[2].button("☆ Favori", key=f"fav_add_list_{i}", use_container_width=True):
                    fav_add(rr); st.toast("Favoriye eklendi", icon="⭐")

            desc = (rr.get("description","") or "").strip()
            if desc:
                with st.expander("Detay (temizlenmiş)", expanded=False):
                    st.write(clean_desc_html(desc))

def page_health(df: pd.DataFrame):
    st.subheader("✅ Sağlık")
    c1,c2,c3 = st.columns(3)
    c1.metric("Toplam ilan", f"{len(df)}")
    c2.metric("Kaynak sayısı", f"{df['source'].nunique() if 'source' in df.columns else '-'}")
    c3.metric("Link OK oranı", f"{df['link_ok'].mean():.2%}" if "link_ok" in df.columns else "-")

    st.divider()
    if "link_reason" in df.columns:
        st.subheader("Link reason dağılımı (Top 10)")
        vc = df["link_reason"].fillna("NA").astype(str).value_counts().head(10).reset_index()
        vc.columns = ["link_reason", "count"]
        render_table_dark(vc, height_px=260)

def page_reco(df: pd.DataFrame):
    st.subheader("🎯 Öneri (baseline)")
    st.caption("Basit skill eşleşmesi: kullanıcının skill seti ile ilandaki skills_str kesişimi.")
    user = st.text_input("Skill’lerin (virgülle)", st.session_state.get("profile",{}).get("skills","python, sql, git"))
    user_sk = set([x.strip().lower() for x in user.split(",") if x.strip()])
    q = st.text_input("Başlık/Şirket (opsiyonel)", "")
    topn = st.slider("Top N", 10, 200, 50, 10)

    d = df.copy()
    if q.strip():
        qq = q.strip().lower()
        d = d[d["title"].str.lower().str.contains(qq, na=False) | d["company"].str.lower().str.contains(qq, na=False)].copy()

    def score_row(r):
        js = set(parse_skills_csv(r.get("skills_str","")))
        if not user_sk: return 0.0
        inter = len(user_sk & js)
        coverage = inter / max(1, len(user_sk))
        precision = inter / max(1, len(js))
        return 0.75*coverage + 0.25*precision

    d["score"] = d.apply(score_row, axis=1)
    d = d.sort_values("score", ascending=False)

    for i, r in d.head(topn).iterrows():
        rr = row_to_dict(r)
        title = (rr["title"] or "").strip() or "(Başlık yok)"
        company = (rr["company"] or "").strip()
        url = (rr["apply_url"] or rr["url"] or "").strip()
        sc = float(r.get("score",0.0))
        js = set(parse_skills_csv(rr.get("skills_str","")))
        matched = sorted(list(user_sk & js))
        missing = sorted(list(js - user_sk))[:12]

        with st.container(border=True):
            top = st.columns([5,1.5,1.5])
            top[0].markdown(f"### {title}")
            top[0].caption(f"**{company}** • skor={sc:.2f}")
            if url:
                try: top[1].link_button("Başvur", url, use_container_width=True)
                except Exception: top[1].markdown(f"[Başvur]({url})")

            k = fav_key(rr)
            if k in fav_store():
                if top[2].button("⭐ Kaldır", key=f"fav_rm_reco_{i}", use_container_width=True):
                    fav_remove(k); st.toast("Favoriden kaldırıldı", icon="⭐")
            else:
                if top[2].button("☆ Favori", key=f"fav_add_reco_{i}", use_container_width=True):
                    fav_add(rr); st.toast("Favoriye eklendi", icon="⭐")

            st.write(f"✅ Eşleşen: {', '.join(matched) if matched else '-'}")
            st.write(f"🧩 Öneri: {', '.join(missing) if missing else '-'}")

def page_model_semantic(df: pd.DataFrame):
    st.subheader("🧠 Model (semantic) – TF-IDF benzerlik demo")
    st.caption("Rol / arama metni gir → ilan metinleriyle semantik benzerlik (cosine).")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        st.error(f"scikit-learn yok: {e}")
        return

    left,_ = st.columns([2,3])
    with left:
        role = st.selectbox("Hedef rol", list(ROLE_PRESETS.keys()) + ["Custom"], index=0)
        custom = st.text_area("Custom arama metni (rol=Custom ise)", "", height=90)
        topn = st.slider("Top N", 10, 200, 50, 10)
        only_ok = st.toggle("Sadece link sağlam", value=True) if "link_ok" in df.columns else False
        sources = sorted(df["source"].astype(str).unique().tolist()) if "source" in df.columns else []
        pick = st.multiselect("Kaynak", sources, default=sources)

    query = custom.strip() if role == "Custom" else ROLE_PRESETS.get(role,"").strip()
    if not query:
        st.warning("Arama metni boş.")
        return

    d = df.copy()
    if pick:
        d = d[d["source"].astype(str).isin(pick)].copy()
    if only_ok and "link_ok" in d.columns:
        d = d[d["link_ok"] == True].copy()

    text = (
        d["title"].fillna("").astype(str) + " " +
        d["company"].fillna("").astype(str) + " " +
        d["location"].fillna("").astype(str) + " " +
        d["skills_str"].fillna("").astype(str) + " " +
        d["description"].fillna("").astype(str).map(lambda x: x[:4000])
    ).fillna("").astype(str).tolist()

    vec = TfidfVectorizer(lowercase=True, max_features=60000, ngram_range=(1,2))
    X = vec.fit_transform(text)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:topn]
    out = d.iloc[idx].copy()
    out["score"] = sims[idx]

    st.caption(f"Aktif veri: {len(d)} satır • Query: {query}")

    for j, (_, r) in enumerate(out.iterrows()):
        rr = row_to_dict(r)
        title = (rr.get("title","") or "").strip() or "(Başlık yok)"
        company = (rr.get("company","") or "").strip()
        url = (rr.get("apply_url","") or rr.get("url","") or "").strip()
        sc = float(r.get("score",0.0))

        with st.container(border=True):
            top = st.columns([5,1.5,1.5])
            top[0].markdown(f"### {title}")
            top[0].caption(f"**{company}** • skor={sc:.3f}")

            if url:
                try: top[1].link_button("Başvur", url, use_container_width=True)
                except Exception: top[1].markdown(f"[Başvur]({url})")

            k = fav_key(rr)
            if k in fav_store():
                if top[2].button("⭐ Kaldır", key=f"fav_rm_sem_{j}", use_container_width=True):
                    fav_remove(k); st.toast("Favoriden kaldırıldı", icon="⭐")
            else:
                if top[2].button("☆ Favori", key=f"fav_add_sem_{j}", use_container_width=True):
                    fav_add(rr); st.toast("Favoriye eklendi", icon="⭐")

def page_profile():
    st.subheader("👤 Profilim")
    st.caption("Skill'lerini kaydet. Career Advisor ve semantic model otomatik kullanır.")
    prof = st.session_state.get("profile", {})
    skills = st.text_input("Skill setin (virgülle)", value=prof.get("skills","python, sql, pandas"))
    role = st.selectbox("Hedef rol", list(ROLE_PRESETS.keys()),
                        index=list(ROLE_PRESETS.keys()).index(prof.get("role","Veri Bilimci")) if prof.get("role","Veri Bilimci") in ROLE_PRESETS else 0)
    city = st.text_input("Şehir/Ülke", value=prof.get("city","Remote"))
    if st.button("Kaydet", use_container_width=True):
        st.session_state["profile"] = {"skills": skills, "role": role, "city": city}
        st.success("Profil kaydedildi ✅")
    p = st.session_state.get("profile", {})
    ca_html(f"""
<div class="ca-card">
  <b>Hedef Rol:</b> {_html.escape(str(p.get("role","—")))}<br/>
  <b>Lokasyon:</b> {_html.escape(str(p.get("city","—")))}<br/>
  <b>Skill set:</b> {_html.escape(str(p.get("skills","—")))}
</div>
""")

def page_cv_career_advisor():
    st.subheader("📄 CV → Career Advisor (PRO)")
    st.caption("PDF → metin çıkar → rol + beceri + eksikler + öğrenme yolu + Knowledge Graph")
    with st.sidebar:
        st.session_state["api_base"] = st.text_input("API Base (FastAPI)", value=api_base())

    up = st.file_uploader("CV PDF yükle", type=["pdf"])
    text = ""
    if up is not None:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(up.getvalue()))
            parts = []
            for p in reader.pages[:25]:
                try: parts.append(p.extract_text() or "")
                except Exception: pass
            text = "\n".join(parts).strip()
        except Exception as e:
            st.error(f"PDF okuma hatası: {e}")
            return
        if not text:
            st.warning("PDF metni çıkarılamadı (scan olabilir).")
            return
        st.success("PDF metni çıkarıldı ✅")
        with st.expander("Çıkan metin (kısaltılmış)", expanded=False):
            st.write(text[:2500])

    if text:
        local_sk = extract_skills_from_text(text, topk=30)
        local_role, _ = infer_role_from_text(text)
        st.markdown("#### Hızlı (lokal) tahmin")
        c1,c2 = st.columns([2,3])
        with c1:
            st.metric("Lokal rol", local_role)
            st.caption("Bu hızlı demo; asıl karar API modelinde.")
        with c2:
            badge_row(local_sk, cls="ca-badge ca-badge-skill", maxn=18)

    go = st.button("🧭 Kariyer Haritamı Çıkar", use_container_width=True, disabled=not bool(text))
    if not go:
        return

    # API varsa kullan, yoksa LOCAL MODE ile devam et
    api_ok = True
    try:
        _ = api_get("/health")
    except Exception:
        api_ok = False

    if not api_ok:
        st.warning("⚠️ API (FastAPI) Cloud'da yok. LOCAL MODE çalışıyor: rol/skill tahmini lokal yapılacak.")
        local_sk = extract_skills_from_text(text, topk=30)
        local_role, _ = infer_role_from_text(text)

        st.divider()
        st.markdown("## 🚀 Sonuç (Local CareerAtlas)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tahmin Rol", local_role)
        c2.metric("Çıkan Skill", str(len(local_sk)))
        c3.metric("API", "Kapalı")

        st.markdown("### ✅ Çıkan beceriler")
        badge_row(local_sk, cls="ca-badge ca-badge-skill", maxn=24)

        st.info("Cloud demo sürümünde FastAPI olmadığı için KG/XAI çalışmaz. İstersen FastAPI'yi ayrıca deploy ederiz.")
        return

    try:
        career = api_post("/career_advisor", {"cv_text": text})
        xai    = api_post("/explain_role",   {"cv_text": text})
        role   = career.get("predicted_role") or career.get("target_role") or "Diğer"
        kgq    = api_post("/kg/query", {"role": role, "top_k": 10})
    except Exception as e:
        st.error(f"API çağrısı başarısız: {e}")
        st.stop()

    st.divider()
    st.markdown("## 🚀 Sonuç (CareerAtlas)")
    top = st.columns([2,2,2])
    top[0].metric("Tahmin Rol", str(role))
    top[1].metric("Fit Score", f"{career.get('fit_score_percent','-')}%")
    top[2].metric("Skill sayısı", f"{len(career.get('extracted_skills',[]) or [])}")

    st.markdown("### ✅ Çıkan beceriler")
    skills = career.get("extracted_skills",[]) or []
    badge_row(skills, cls="ca-badge ca-badge-skill", maxn=24)
    if not skills:
        st.caption("Skill çıkmadı.")

    st.markdown("### 🧩 Eksik beceriler (öncelikli)")
    missp = career.get("missing_skills_prioritized",[]) or []
    badge_row([x.get("skill","") for x in missp[:20]], cls="ca-badge ca-badge-miss", maxn=20)
    if not missp:
        st.caption("Eksik skill listesi yok.")

    st.markdown("### 🔎 Neden bu rol? (XAI)")
    pred = xai.get("predicted_role", role) if isinstance(xai, dict) else role
    x_sk = xai.get("extracted_skills", []) if isinstance(xai, dict) else []
    ca_html(f"""
<div class="ca-card">
  <b>Model rol tahmini:</b> {_html.escape(str(pred))}<br/>
  <span style="color:var(--muted)">Bu bölüm ekranda JSON basmaz; indirilebilir çıktı verir.</span>
</div>
""")
    if x_sk:
        badge_row(x_sk, cls="ca-badge ca-badge-skill", maxn=20)
    st.download_button(
        "⬇️ XAI JSON indir",
        data=json.dumps(xai, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="xai.json",
        mime="application/json",
        use_container_width=True
    )

    st.markdown("### 🕸️ Knowledge Graph (Role ↔ Skill ↔ Course)")
    nodes = kgq.get("nodes") if isinstance(kgq, dict) else None
    edges = kgq.get("edges") if isinstance(kgq, dict) else None
    if nodes and edges:
        render_kg_pyvis(nodes, edges, height=560)
    else:
        ca_html("""<div class="ca-card">KG graf formatında gelmemiş. Ekranda JSON göstermiyorum. Aşağıdan indir.</div>""")
    st.download_button(
        "⬇️ KG JSON indir",
        data=json.dumps(kgq, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="kg.json",
        mime="application/json",
        use_container_width=True
    )

_INTERVIEW_QS = {
 "Veri Bilimci": [
   "Overfitting nedir, nasıl önlersin? (En az 3 yöntem)",
   "A/B testinde p-value ve power ne demek? Hangi durumda yanlış karar verirsin?",
   "Bir modeli production'a alırken hangi metrikleri izlersin? Drift'i nasıl anlarsın?",
 ],
 "Veri Mühendisi": [
   "Batch vs Streaming: trade-off’lar neler? Ne zaman hangisi?",
   "ETL ile ELT farkı nedir? Ne zaman hangisi?",
 ],
 "Backend Engineer": [
   "Bir REST API'yi ölçeklemek için neler yaparsın? Bottleneck'i nasıl bulursun?",
   "Cache stratejileri: write-through vs write-back farkı nedir?",
 ],
 "SRE / DevOps": [
   "Sev-1 incidentte ilk 10 dakikada ne yaparsın?",
   "SLO/SLI/SLA farkı nedir? Örnekle anlat.",
 ],
}

def _star_parts(answer: str) -> dict:
    t = (answer or "").lower()
    parts = {
      "Situation / Durum": any(k in t for k in ["situation","durum","bağlam"]),
      "Task / Görev": any(k in t for k in ["task","görev","hedef"]),
      "Action / Aksiyon": any(k in t for k in ["action","aksiyon","uygulad","yapt","implement","kur","tasarla","optimiz"]),
      "Result / Sonuç": any(k in t for k in ["result","sonuç","çıktı","%","iyileş","azal","art","latency","maliyet","gelir"]),
    }
    score = int(25 * sum(1 for v in parts.values() if v))
    return {"parts": parts, "score": score}

def _simple_feedback(answer: str, role: str, skills_csv: str):
    a = (answer or "").strip()
    if not a:
        return 0, ["Cevap boş. En az 5-6 cümle yaz."], []
    tips=[]; good=[]; score=50
    if len(a) >= 420: score += 15; good.append("Detay seviyesi iyi (uzunluk).")
    elif len(a) >= 240: score += 8; good.append("Yeterli uzunluk.")
    else: tips.append("Biraz kısa. 1 örnek + adım ekle (STAR: Durum-Görev-Aksiyon-Sonuç).")

    kw = set([t for t in ROLE_PRESETS.get(role,"").lower().split() if len(t)>=3])
    hit = sum(1 for k in list(kw)[:30] if k in a.lower())
    if hit >= 6: score += 15; good.append("Rol anahtar kelimeleri iyi.")
    else: tips.append("Rol anahtar kelimeleri ekle: " + ", ".join(list(kw)[:8]))

    skills = [x.strip().lower() for x in (skills_csv or "").split(",") if x.strip()]
    sh = sum(1 for s in skills[:20] if s in a.lower())
    if sh >= 2: score += 10; good.append("Kendi skill'lerini referanslamışsın.")
    else: tips.append("Cevaba kendi skill’lerinden 2-3 tanesini iliştir (ör: python, sql, docker…).")

    score=max(0,min(100,score))
    return score, tips, good

def _followup(answer: str, role: str, skills_csv: str) -> str:
    a = (answer or "").lower()
    skills = [x.strip().lower() for x in (skills_csv or "").split(",") if x.strip()]
    for s in skills[:12]:
        if s and s in a:
            return f"'{s}' kullandığın bölümde en büyük zorluk neydi ve nasıl çözdün?"
    preset = (ROLE_PRESETS.get(role,"") or "").split()
    for tok in preset[:20]:
        if tok in a:
            return f"'{tok}' ile ilgili trade-off neydi? Alternatifleri neden elemiştin?"
    return "Sonucu bir metrikle netleştir: % iyileşme / süre azalma / maliyet düşüşü / gelir artışı gibi."

def page_interview():
    st.subheader("🎤 Mülakat Simülasyonu – İnteraktif")
    st.caption("Soru seç → cevap yaz → skor + STAR checklist + takip sorusu ✅")
    prof = st.session_state.get("profile", {})
    role = prof.get("role","Veri Bilimci")
    skills = prof.get("skills","python, sql, pandas")

    c1,c2 = st.columns([2,3])
    with c1:
        role2 = st.selectbox("Rol", list(ROLE_PRESETS.keys()), index=list(ROLE_PRESETS.keys()).index(role) if role in ROLE_PRESETS else 0)
        skills2 = st.text_input("Skill set (profil)", value=skills)
        show_rubric = st.toggle("STAR ipucu göster", value=True)
        st.session_state["profile"] = {"skills": skills2, "role": role2, "city": prof.get("city","")}
    with c2:
        if show_rubric:
            ca_html("""<div class="ca-card"><b>STAR ipucu</b><br/>
            <span style="color:var(--muted)">Situation → Task → Action → Result. En az 1 metrik ekle (latency %30 azaldı gibi).</span>
            </div>""")

    qlist = _INTERVIEW_QS.get(role2, _INTERVIEW_QS["Veri Bilimci"])
    q = st.selectbox("Soru seç", qlist, index=0)
    ans = st.text_area("Cevabın", height=180, placeholder="Cevabını STAR formatında yaz…")

    if "interview_history" not in st.session_state:
        st.session_state["interview_history"] = []

    if st.button("🧪 Değerlendir", use_container_width=True):
        base, tips, good = _simple_feedback(ans, role2, skills2)
        star = _star_parts(ans)
        bonus = int(min(20, (star["score"]/100)*20))
        total = int(max(0, min(100, base + bonus)))

        st.progress(total/100.0)
        st.metric("Skor", f"{total}/100")

        cols = st.columns(4)
        for i,(k,v) in enumerate(star["parts"].items()):
            with cols[i]:
                st.write(("✅ " if v else "⬜ ") + k)

        st.info("Takip sorusu: " + _followup(ans, role2, skills2))
        if good:
            st.success("İyi noktalar:\n- " + "\n- ".join(good))
        if tips:
            st.warning("Geliştirme önerileri:\n- " + "\n- ".join(tips))

        st.session_state["interview_history"].insert(0, {
            "role": role2, "q": q, "score": total,
            "preview": (ans[:220] + "…") if len(ans or "")>220 else (ans or "")
        })
        st.session_state["interview_history"] = st.session_state["interview_history"][:10]

    with st.expander("🕘 Geçmiş (son 10)", expanded=False):
        if not st.session_state["interview_history"]:
            st.caption("Henüz deneme yok.")
        for h in st.session_state["interview_history"]:
            st.write(f"**{h['score']}/100** • {h['role']} • {h['q']}")
            st.caption(h["preview"])

def page_favorites():
    st.subheader("⭐ Favoriler")
    fav = fav_store()
    st.caption(f"Toplam: **{len(fav)}**")
    if not fav:
        st.info("Henüz favori yok. İlanlar/Model/Öneri sayfalarından ☆ Favori ile ekle.")
        return

    rows=[]
    for k,v in fav.items():
        rows.append({"key": k, **v})
    fdf = pd.DataFrame(rows)
    show = fdf.drop(columns=["key"]) if "key" in fdf.columns else fdf
    render_table_dark(show, height_px=360)

    c1,c2,c3 = st.columns([2,2,2])
    with c1:
        if st.button("🧹 Tüm favorileri temizle", use_container_width=True):
            st.session_state["favorites"] = {}
            st.success("Favoriler temizlendi.")
            st.rerun()
    with c2:
        sel = st.selectbox("Silmek için seç", list(fav.keys()))
        if st.button("🗑️ Seçileni sil", use_container_width=True):
            fav_remove(sel)
            st.success("Silindi.")
            st.rerun()
    with c3:
        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV indir", data=csv, file_name="favorites.csv", mime="text/csv", use_container_width=True)

def page_twin():
    st.subheader("🧬 Dijital İkiz (Demo)")
    prof = st.session_state.get("profile", {})
    fav  = fav_store()

    c1,c2 = st.columns([2,3])
    with c1:
        ca_html(f"""
<div class="ca-card"><b>Profil</b><br/>
  <span style="color:var(--muted)"><b>Hedef Rol:</b> {_html.escape(str(prof.get("role","—")))}</span><br/>
  <span style="color:var(--muted)"><b>Lokasyon:</b> {_html.escape(str(prof.get("city","—")))}</span><br/>
  <span style="color:var(--muted)"><b>Skill set:</b> {_html.escape(str(prof.get("skills","—")))}</span>
</div>
""")
        ca_html(f"""<div class="ca-card" style="margin-top:12px;"><b>Favoriler</b><br/>⭐ <b>{len(fav)}</b> ilan</div>""")
        if fav:
            st.caption("Son eklenenler:")
            for it in list(fav.values())[:8]:
                t = it.get("title","-"); u = it.get("url","")
                if u: st.markdown(f"- [{t}]({u})")
                else: st.write(f"- {t}")

    with c2:
        ca_html("""
<div class="ca-card">
  <b>Sunum cümlesi:</b><br/>
  “CV’den rol/skill çıkarıp ilanlara eşleştiriyor, learning path veriyor ve graf üstünden kurs/skill ilişkisi gösteriyor.”
  <br/><br/><span style="color:var(--muted)">Not: Bu sayfada JSON/kod bloğu göstermiyoruz (ürün hissi).</span>
</div>
""")

def main():
    st.set_page_config(page_title="CareerAtlas", layout="wide")
    inject_css()

    p = pick_dataset()
    if p is None:
        st.error("Dataset bulunamadı: data/app veya data/derived altına bak.")
        st.stop()

    st.session_state["dataset_path"] = str(p)
    df = load_df(str(p))

    with st.sidebar:
        ca_html(f"""
<div class="ca-card" style="padding:12px 14px; margin-top:6px; margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:10px;">
    {COMPASS_SVG}
    <div>
      <div style="font-weight:900;font-size:18px;letter-spacing:-0.02em;">CareerAtlas</div>
      <div style="color:var(--muted);font-size:12px;">Navigate your career</div>
    </div>
  </div>
</div>
""")
        menu = st.radio("Menü", [
          "🏠 Home",
          "📄 CV → Career Advisor",
          "🧠 Model (semantic)",
          "🗂️ İlanlar",
          "✅ Sağlık",
          "🎯 Öneri (baseline)",
          "👤 Profilim",
          "🎤 Mülakat",
          "⭐ Favoriler",
          "🧬 Dijital İkiz",
        ], index=0)

    if menu == "🏠 Home":
        page_home(df)
    elif menu == "📄 CV → Career Advisor":
        page_cv_career_advisor()
    elif menu == "🧠 Model (semantic)":
        page_model_semantic(df)
    elif menu == "🗂️ İlanlar":
        page_listings(df)
    elif menu == "✅ Sağlık":
        page_health(df)
    elif menu == "🎯 Öneri (baseline)":
        page_reco(df)
    elif menu == "👤 Profilim":
        page_profile()
    elif menu == "🎤 Mülakat":
        page_interview()
    elif menu == "⭐ Favoriler":
        page_favorites()
    else:
        page_twin()

if __name__ == "__main__":
    main()










