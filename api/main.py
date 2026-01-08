from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
import re

app = FastAPI(title="CareerAtlas API")

class CVReq(BaseModel):
    cv_text: str

class RoleReq(BaseModel):
    role: str
    top_k: int = 10

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
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_skills_from_text(text: str, topk: int = 30) -> List[str]:
    tx = normalize_text(text)
    hits = []
    for sk in SKILL_VOCAB:
        if sk.lower() in tx:
            hits.append("scikit-learn" if sk == "sklearn" else sk)
    out=[]
    seen=set()
    for h in hits:
        hh=h.lower()
        if hh not in seen:
            seen.add(hh)
            out.append(h)
    return out[:topk]

def infer_role_from_text(text: str):
    tx = normalize_text(text)
    scores={}
    for role,q in ROLE_PRESETS.items():
        toks=[t.strip().lower() for t in q.split() if t.strip()]
        scores[role]=sum(1 for t in toks if t in tx)
    best=max(scores, key=scores.get) if scores else "Veri Bilimci"
    return best, scores

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/career_advisor")
def career_advisor(req: CVReq) -> Dict[str, Any]:
    skills = extract_skills_from_text(req.cv_text, topk=30)
    role, scores = infer_role_from_text(req.cv_text)

    # Basit "missing" üretelim: preset'teki terimlerden cv'de yoksa eksik say
    preset = ROLE_PRESETS.get(role, "")
    preset_terms = [t.strip().lower() for t in preset.split() if t.strip()]
    tx = normalize_text(req.cv_text)
    missing = []
    for t in preset_terms:
        if t not in tx and len(t) >= 3:
            missing.append({"skill": t, "priority": 1})
    missing = missing[:15]

    return {
        "predicted_role": role,
        "fit_score_percent": 60 + min(35, 3*len(skills)),
        "extracted_skills": skills,
        "missing_skills_prioritized": missing,
        "role_scores": scores
    }

@app.post("/explain_role")
def explain_role(req: CVReq) -> Dict[str, Any]:
    skills = extract_skills_from_text(req.cv_text, topk=30)
    role, scores = infer_role_from_text(req.cv_text)
    return {
        "predicted_role": role,
        "extracted_skills": skills,
        "role_scores": scores,
        "explanation": "Lokal heuristic: rol preset anahtar kelimeleri ile eşleşme skorları."
    }

@app.post("/kg/query")
def kg_query(req: RoleReq) -> Dict[str, Any]:
    # Demo Knowledge Graph: role -> skill -> course
    role = req.role or "Role"
    preset = ROLE_PRESETS.get(role, "")
    skills = [t.strip() for t in preset.split() if t.strip()][: req.top_k]

    nodes=[]
    edges=[]
    nodes.append({"id": f"role:{role}", "label": role, "type": "role"})

    for s in skills:
        sid = f"skill:{s}"
        nodes.append({"id": sid, "label": s, "type": "skill"})
        edges.append({"source": f"role:{role}", "target": sid, "type": "requires"})

        # Basit course node
        cid = f"course:{s}"
        nodes.append({"id": cid, "label": f"{s} 101", "type": "course"})
        edges.append({"source": sid, "target": cid, "type": "learn"})

    return {"nodes": nodes, "edges": edges}
