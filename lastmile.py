# lastmile2.py — Last Mile - Loja Perfeita (Login + Admin + MySQL + IA + Mapa + Catálogo + Dashboard com Período)
# ---------------------------------------------------------------------------------------------------------------
# - LOGIN (usuário/senha) e sessão
# - ADMIN: cria usuários (admin/promotor) e vincula promotor a filial + id/nome
# - PROMOTOR logado: vê apenas a própria filial e os seus clientes do dia
# - Fluxo: Check-in → Gôndola → Estoque → Checklist (TTS/voz) → Revisão Final → Dashboard
# - Banco: MySQL (roteiros) + SQLite (visits + users, com migrações)
# - IA: análises visão, TTS robusto (texto "falável"), STT, % vazio e comparação final
# - Catálogo: informe os seus produtos (texto/CSV/Excel) para IA reconhecer só “os seus”
# - Dashboard com seletor de período: Hoje / Semana / Mês / Tudo
# - .env: OPENAI_API_KEY, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, ADMIN_PASSWORD (opcional)
# ---------------------------------------------------------------------------------------------------------------

import os, io, re, json, base64, sqlite3, tempfile, datetime, hashlib, secrets
from typing import Optional, List, Tuple
import pandas as pd
import streamlit as st
from PIL import Image

# ===== .env =====
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Config Streamlit =====
st.set_page_config(page_title="Last Mile — Loja Perfeita", page_icon="🛒", layout="wide")

# ===== MySQL (sem ODBC) =====
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

MYSQL_HOST = os.getenv("MYSQL_HOST", "10.40.13.6")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "cassio")
MYSQL_PWD_PLAIN = os.getenv("MYSQL_PASSWORD", "")
MYSQL_PWD = quote_plus(MYSQL_PWD_PLAIN)

@st.cache_resource
def get_engine_no_db():
    uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/?charset=utf8mb4"
    return create_engine(uri, pool_pre_ping=True)

def get_engine_for_schema(schema: str):
    uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/{schema}?charset=utf8mb4"
    return create_engine(uri, pool_pre_ping=True)

# Filiais e schemas
FILIAIS = ["ABC","API","MCD","TBE","TBL","TCA","TCG","TCV","TPA","TPH","TSJ"]
FILIAL_TO_SCHEMA = {
    "ABC":"promoter_abc","API":"promoter_api","MCD":"promoter_mcd",
    "TBE":"promoter_tbe","TBL":"promoter_tbl","TCA":"promoter_tca",
    "TCG":"promoter_tcg","TCV":"promoter_tcv","TPA":"promoter_tpa",
    "TPH":"promoter_tph","TSJ":"promoter_tsj"
}
def schema_from_filial(filial: str) -> str:
    return FILIAL_TO_SCHEMA.get(filial.upper(), f"promoter_{filial.lower()}")

# ===== Detecção tolerante de colunas em `roteiro` =====
def _map_roteiro_columns(conn, schema: str) -> dict:
    rows = conn.execute(text(f"SHOW COLUMNS FROM `{schema}`.`roteiro`")).fetchall()
    cols = {r[0].lower(): r[0] for r in rows}
    find = lambda cands: next((cols[c] for c in cands if c in cols), None)
    id_col      = find(["id","id_roteiro"]) or "id"
    cliente_col = find(["id_cliente","cliente_id","id_cliente_fk","cliente"]) or "id_cliente"
    rep_col     = find(["id_promotor","id_representante","id_vendedor","id_usuario"])   # opcional
    data_col    = find(["data_visita","data","dt_visita"]) or "data_visita"
    status_col  = find(["status","situacao"]) or "status"
    real_col    = find(["realizado","dt_realizado","data_realizacao"])  # opcional
    return {"id":id_col,"cliente":cliente_col,"rep":rep_col,"data":data_col,"status":status_col,"real":real_col}

# ===== Consultas MySQL (HOJE) =====
def fetch_promoters_today(engine_schema) -> pd.DataFrame:
    q = text("""
        SELECT 
            DATE(`data`) AS data,
            CAST(id_promotor AS CHAR) AS id_promotor,
            CAST(nome_promotor AS CHAR) AS nome_promotor,
            SUM(roteiros_cadastrados) AS roteiros_cadastrados
        FROM `bi_roteiro`
        WHERE DATE(`data`) = CURDATE()
        GROUP BY DATE(`data`), id_promotor, nome_promotor
        ORDER BY nome_promotor
    """)
    try:
        with engine_schema.connect() as conn:
            return pd.read_sql(q, conn)
    except Exception:
        return pd.DataFrame()

def fetch_today_routes_with_client(engine_schema, id_promotor: Optional[str]=None) -> pd.DataFrame:
    with engine_schema.connect() as conn:
        m = _map_roteiro_columns(conn, engine_schema.url.database)
        rep_sel  = f"r.`{m['rep']}`"  if m["rep"]  else "NULL"
        real_sel = f"r.`{m['real']}`" if m["real"] else "NULL"
        base_sql = f"""
        SELECT 
            r.`{m['id']}`          AS id,
            r.`{m['cliente']}`     AS id_cliente,
            {rep_sel}              AS id_promotor,
            r.`{m['data']}`        AS data_visita,
            r.`{m['status']}`      AS status,
            {real_sel}             AS realizado,
            c.fantasia             AS nome_cliente,
            c.razao_social,
            c.latitude,
            c.longitude
        FROM `roteiro` r
        LEFT JOIN `cliente` c ON c.id = r.`{m['cliente']}`
        WHERE DATE(r.`{m['data']}`) = CURDATE()
        """
        params = {}
        if id_promotor and m["rep"]:
            base_sql += " AND r.`{rep}` = :pid".format(rep=m["rep"])
            params["pid"] = id_promotor
        base_sql += f" ORDER BY r.`{m['data']}`, r.`{m['id']}`"
        df = pd.read_sql(text(base_sql), conn, params=params)

    if not df.empty:
        for c in ["id","id_cliente","id_promotor","status"]:
            if c in df.columns: df[c] = df[c].astype(str)
        if "data_visita" in df.columns:
            df["data_visita"] = pd.to_datetime(df["data_visita"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        df["visitado"] = df.get("realizado").notna() if "realizado" in df.columns else False
        df["nome_cliente"] = df.get("nome_cliente").fillna(df.get("razao_social"))
    return df

# ===== OpenAI (IA) =====
CLIENT = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

API_KEY_FIXA = os.getenv("OPENAI_API_KEY", "").strip()
def ensure_openai_client():
    global CLIENT
    if CLIENT: return CLIENT
    if not OpenAI: return None
    key = API_KEY_FIXA or os.getenv("OPENAI_API_KEY","").strip()
    if not key: return None
    CLIENT = OpenAI(api_key=key)
    return CLIENT

def image_to_b64_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    pic = img.copy(); pic.thumbnail((1200,1200))
    pic.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def gpt_vision_multi(prompt: str, images: List[Image.Image], max_tokens=900) -> str:
    client = ensure_openai_client()
    if not client: return "⚠️ IA desligada (defina OPENAI_API_KEY)."
    content = [{"type":"text","text":prompt}] + [
        {"type":"image_url","image_url":{"url":image_to_b64_url(i)}} for i in images
    ]
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.3, max_tokens=max_tokens,
            messages=[{"role":"system","content":"Você é um consultor de execução em PDV, direto e prático."},
                      {"role":"user","content":content}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Falha IA: {e}"

# === Voz (microfone) opcional
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_OK = True
except Exception:
    MIC_OK = False

# ======= Sanitização para TTS (não ler símbolos/markdown) =======
def _speakable_pt(txt: str) -> str:
    if not txt: return ""
    t = txt
    t = re.sub(r'^\s{0,3}#{1,6}\s*', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*[-•▪◦]\s*', '', t, flags=re.MULTILINE)
    t = re.sub(r'\*\*(.*?)\*\*', r'\1', t)
    t = re.sub(r'\*(.*?)\*', r'\1', t)
    t = re.sub(r'`([^`]*)`', r'\1', t)
    repl = {'%':' por cento','&':' e ','/':' barra ','+':' mais ','@':' arroba ',
            '–':'-','—':'-','•':' ','►':' ','»':' ','«':' '}
    for k,v in repl.items(): t = t.replace(k, v)
    t = re.sub(r'(\d+)\s*barra\s*(\d+)', r'\1 de \2', t)
    t = re.sub(r'[{}[\]<>_=\\|~^]+', ' ', t)
    t = re.sub(r'\s+',' ', t).strip()
    return t[:800]

def tts_bytes(texto: str):
    texto = _speakable_pt(texto or "")
    client = ensure_openai_client()
    if not client: return b"", "audio/mp3"
    try:
        resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=texto)
        if hasattr(resp,"read"): return resp.read(), "audio/mp3"
        if hasattr(resp,"content"): return resp.content, "audio/mp3"
        return bytes(resp), "audio/mp3"
    except TypeError:
        try:
            resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=texto, response_format="mp3")
            if hasattr(resp,"read"): return resp.read(), "audio/mp3"
            if hasattr(resp,"content"): return resp.content, "audio/mp3"
            return bytes(resp), "audio/mp3"
        except Exception: pass
    except Exception: pass
    try:
        with client.audio.speech.with_streaming_response.create(model="gpt-4o-mini-tts", voice="alloy", input=texto) as stream:
            return stream.read(), "audio/mp3"
    except Exception as e:
        st.warning(f"TTS (OpenAI) falhou: {e}")
        return b"", "audio/mp3"

def stt_from_bytes(audio_bytes: bytes) -> str:
    client = ensure_openai_client()
    if not client: return ""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes); path = tmp.name
    try:
        with open(path, "rb") as af:
            tr = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=af)
        return (tr.text or "").strip().lower()
    except Exception:
        return ""

# ===== % Vazio =====
def _estimate_empty_percent(img: Image.Image):
    client = ensure_openai_client()
    if not client: return None
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=10,
            messages=[{"role":"user","content":[
                {"type":"text","text":"Responda apenas um número 0..100: % de área VAZIA na gôndola."},
                {"type":"image_url","image_url":{"url": image_to_b64_url(img)}}
            ]}]
        )
        m = re.search(r"(\d{1,3})", (r.choices[0].message.content or ""))
        if not m: return None
        return max(0, min(100, int(m.group(1))))
    except Exception:
        return None

def estimate_empty_before_after(before_img: Image.Image, after_img: Image.Image):
    return _estimate_empty_percent(before_img), _estimate_empty_percent(after_img)

# ===== SQLite (visits + users) =====
DB_PATH = os.path.join(os.path.dirname(__file__), "lastmile.db")
def sql_execute(q, params=()):
    con = sqlite3.connect(DB_PATH); cur = con.cursor(); cur.execute(q, params); con.commit(); con.close()
def sql_fetch_df(q, params=()):
    con = sqlite3.connect(DB_PATH); df = pd.read_sql_query(q, con, params=params); con.close(); return df

# --- Hash de senha (bcrypt -> fallback PBKDF2) ---
try:
    import bcrypt
    def _hash_pw(pw: str) -> str:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    def _verify_pw(pw: str, h: str) -> bool:
        try: return bcrypt.checkpw(pw.encode(), h.encode())
        except Exception: return False
except Exception:
    def _hash_pw(pw: str) -> str:
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200000).hex()
        return f"pbkdf2${salt}${dk}"
    def _verify_pw(pw: str, h: str) -> bool:
        try:
            _, salt, dk = h.split("$", 2)
            ndk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200000).hex()
            return secrets.compare_digest(ndk, dk)
        except Exception: return False

def init_db():
    # visits
    sql_execute("""
    CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        loja TEXT,
        filial TEXT,
        checkin_ts TEXT
    )""")
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("PRAGMA table_info(visits)")
    have = {r[1] for r in cur.fetchall()}
    def need(col, ddl):
        if col not in have: cur.execute(ddl)
    need("id_cliente",      "ALTER TABLE visits ADD COLUMN id_cliente TEXT")
    need("promotor_id",     "ALTER TABLE visits ADD COLUMN promotor_id TEXT")
    need("promotor_nome",   "ALTER TABLE visits ADD COLUMN promotor_nome TEXT")
    need("lat",             "ALTER TABLE visits ADD COLUMN lat REAL")
    need("lon",             "ALTER TABLE visits ADD COLUMN lon REAL")
    need("checkout_ts",     "ALTER TABLE visits ADD COLUMN checkout_ts TEXT")
    need("duracao_min",     "ALTER TABLE visits ADD COLUMN duracao_min REAL")
    need("eficiencia",      "ALTER TABLE visits ADD COLUMN eficiencia REAL")
    need("analysis_gondola","ALTER TABLE visits ADD COLUMN analysis_gondola TEXT")
    need("analysis_estoque","ALTER TABLE visits ADD COLUMN analysis_estoque TEXT")
    need("checklist_json",  "ALTER TABLE visits ADD COLUMN checklist_json TEXT")
    need("result_json",     "ALTER TABLE visits ADD COLUMN result_json TEXT")
    need("catalogo_json",   "ALTER TABLE visits ADD COLUMN catalogo_json TEXT")
    con.commit()

    # users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        pass_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        filial TEXT,
        promotor_id TEXT,
        promotor_nome TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TEXT
    )
    """)
    con.commit()

    # seed admin se não existir
    df_admin = pd.read_sql_query("SELECT id FROM users WHERE username='admin'", con)
    if df_admin.empty:
        admin_pw = os.getenv("ADMIN_PASSWORD", "admin123")
        cur.execute(
            "INSERT INTO users (username, pass_hash, role, is_active, created_at) VALUES (?,?,?,?,?)",
            ("admin", _hash_pw(admin_pw), "admin", 1, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        con.commit()
    con.close()
init_db()

# --- helpers users ---
def user_get(username: str) -> Optional[dict]:
    df = sql_fetch_df("SELECT * FROM users WHERE username=?", (username,))
    return df.to_dict("records")[0] if not df.empty else None
def user_list() -> pd.DataFrame:
    return sql_fetch_df("SELECT id, username, role, filial, promotor_id, promotor_nome, is_active, created_at FROM users ORDER BY username")
def user_create(username: str, password: str, role: str, filial: Optional[str], promotor_id: Optional[str], promotor_nome: Optional[str]) -> Tuple[bool, str]:
    if not username or not password: return False, "Usuário e senha obrigatórios."
    if role not in ("admin","promotor"): return False, "Perfil inválido."
    if role == "promotor" and not (filial and promotor_id and promotor_nome):
        return False, "Para promotor: filial, id_promotor e nome_promotor são obrigatórios."
    try:
        sql_execute(
            "INSERT INTO users (username, pass_hash, role, filial, promotor_id, promotor_nome, is_active, created_at) VALUES (?,?,?,?,?,?,1,?)",
            (username, _hash_pw(password), role, filial, promotor_id, promotor_nome, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        return True, "Usuário criado."
    except sqlite3.IntegrityError:
        return False, "Usuário já existe."
def user_set_active(user_id: int, active: bool):
    sql_execute("UPDATE users SET is_active=? WHERE id=?", (1 if active else 0, user_id))
def user_reset_password(user_id: int, new_pw: str):
    sql_execute("UPDATE users SET pass_hash=? WHERE id=?", (_hash_pw(new_pw), user_id))

# ======= Sessão =======
def ss_init():
    preset = {
        "screen":"login",
        "auth": None,  # {id, username, role, filial, promotor_id, promotor_nome}
        "filial":"", "schema":"",
        "promotor_id":"", "promotor_nome":"",
        "loja_id":"", "loja_nome":"",
        "active_visit_id":None,
        "imgs_gondola":[], "imgs_estoque":[],
        "analysis_gondola":"", "analysis_estoque":"",
        "guided":{"steps":[], "idx":0},
        "final_gondola":None, "final_review_text":"", "final_review_level":"",
        "empty_before":None, "empty_after":None, "empty_delta":None,
        "cam_gondola_key":0, "cam_estoque_key":0, "cam_final_key":0,
        "df_routes_today":pd.DataFrame(),
        "catalogo_produtos":[]  # << lista por visita
    }
    for k,v in preset.items(): st.session_state.setdefault(k,v)
ss_init()
def go(x): st.session_state.screen = x
def require_auth():
    if not st.session_state.auth:
        go("login"); st.stop()

# ====== LOGIN (com rerun) ======
def screen_login():
    if st.session_state.get("auth"):
        go("home"); st.rerun()
    st.markdown("## 🔐 Login")
    with st.form("f_login", clear_on_submit=False):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        ok = st.form_submit_button("Entrar", type="primary", use_container_width=True)
    if ok:
        urow = user_get(u)
        if not urow or not urow.get("is_active"):
            st.error("Usuário inexistente ou inativo."); return
        if not _verify_pw(p, urow["pass_hash"]):
            st.error("Senha inválida."); return
        st.session_state.auth = {
            "id": urow["id"],
            "username": urow["username"],
            "role": urow["role"],
            "filial": urow.get("filial") or "",
            "promotor_id": urow.get("promotor_id") or "",
            "promotor_nome": urow.get("promotor_nome") or "",
        }
        st.session_state.screen = "home"; st.rerun()

# ====== ADMIN ======
def screen_admin():
    require_auth()
    if st.session_state.auth["role"] != "admin":
        st.error("Acesso restrito ao Administrador."); return
    st.markdown("## 🛠️ Painel do Administrador")
    tabs = st.tabs(["👤 Usuários", "➕ Novo usuário"])
    with tabs[0]:
        df = user_list()
        if not df.empty:
            df_show = df.copy()
            df_show["is_active"] = df_show["is_active"].map({1:"Ativo",0:"Inativo"})
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            st.markdown("### Ações rápidas")
            col1, col2, col3 = st.columns(3)
            with col1:
                uid = st.number_input("ID usuário", min_value=1, step=1)
            with col2:
                if st.button("Ativar"): user_set_active(int(uid), True); st.rerun()
                if st.button("Inativar"): user_set_active(int(uid), False); st.rerun()
            with col3:
                nova = st.text_input("Nova senha")
                if st.button("Resetar senha"):
                    if nova: user_reset_password(int(uid), nova); st.success("Senha atualizada."); st.rerun()
                    else: st.warning("Informe a nova senha.")
    with tabs[1]:
        with st.form("f_new_user"):
            colA, colB = st.columns(2)
            with colA:
                username = st.text_input("Usuário")
                role = st.selectbox("Perfil", ["promotor","admin"], index=0)
                senha = st.text_input("Senha temporária", type="password")
            with colB:
                filial_sel = st.selectbox("Filial (para promotor)", FILIAIS, index=0)
                schema = schema_from_filial(filial_sel)
                engine_schema = get_engine_for_schema(schema)
                df_prom = fetch_promoters_today(engine_schema)
                promotor_id = ""; promotor_nome = ""
                if not df_prom.empty:
                    df_prom = df_prom.sort_values("nome_promotor")
                    labels = df_prom.apply(lambda r: f"{r['nome_promotor']} (id {r['id_promotor']})", axis=1).tolist()
                    idxs = list(range(len(labels)))
                    idx = st.selectbox("Promotor do dia (opcional p/ preencher)", idxs, format_func=lambda i: labels[i])
                    row = df_prom.iloc[idx]; promotor_id = str(row["id_promotor"]); promotor_nome = str(row["nome_promotor"])
                promotor_id = st.text_input("ID do promotor", value=promotor_id)
                promotor_nome = st.text_input("Nome do promotor", value=promotor_nome)
            ok = st.form_submit_button("Cadastrar", type="primary")
        if ok:
            filial_val = filial_sel if role=="promotor" else None
            pid = promotor_id if role=="promotor" else None
            pnm = promotor_nome if role=="promotor" else None
            ok, msg = user_create(username, senha, role, filial_val, pid, pnm)
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()

# ===== utilitários fotos / tempo =====
def file_md5(b: bytes) -> str: return hashlib.md5(b).hexdigest()
def add_captured_photo(key: str, file):
    if not file: return
    b = file.getvalue(); md5 = file_md5(b)
    arr = st.session_state.get(key, [])
    if md5 not in [x["md5"] for x in arr]:
        arr.append({"md5":md5,"bytes":b}); st.session_state[key] = arr
def photos_to_pil(key: str) -> List[Image.Image]:
    out=[]
    for ph in st.session_state.get(key, []):
        try: out.append(Image.open(io.BytesIO(ph["bytes"])).convert("RGB"))
        except Exception: pass
    return out
def minutes_between(ts1, ts2):
    try:
        t1=datetime.datetime.strptime(ts1,"%Y-%m-%d %H:%M:%S"); t2=datetime.datetime.strptime(ts2,"%Y-%m-%d %H:%M:%S")
        return round((t2-t1).total_seconds()/60.0,2)
    except Exception: return None

# ====== NOVO: parser do catálogo (textarea + CSV/Excel) ======
def parse_catalog_input(text: str, uploaded_file) -> List[str]:
    items = []
    if text:
        items += [re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if x.strip()]
    if uploaded_file is not None:
        try:
            name = (uploaded_file.name or "").lower()
            if name.endswith(".csv"):
                import csv, io as _io
                content = uploaded_file.getvalue().decode("utf-8", "ignore")
                reader = csv.reader(_io.StringIO(content))
                for row in reader:
                    if row and str(row[0]).strip():
                        items.append(str(row[0]).strip())
            elif name.endswith(".xlsx") or name.endswith(".xls"):
                _df = pd.read_excel(uploaded_file)
                col = _df.columns[0]
                items += [str(x).strip() for x in _df[col].dropna().tolist()]
            else:
                st.warning("Formato não suportado. Use CSV, XLSX ou XLS.")
        except Exception as e:
            st.warning(f"Falha ao ler arquivo: {e}")
    uniq=[]
    for it in items:
        if it and it not in uniq:
            uniq.append(it)
    return uniq[:200]

# ===================== DASHBOARD HELPERS (Período) =====================
def _period_bounds(periodo: str) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    today = datetime.date.today()
    if periodo == "Hoje":
        return today, today
    if periodo == "Semana":
        start = today - datetime.timedelta(days=today.weekday())  # segunda-feira
        return start, today
    if periodo == "Mês":
        start = today.replace(day=1)
        return start, today
    return None, None  # Tudo

def visits_period_df(periodo: str, filial: str | None = None, promotor_id: str | None = None) -> pd.DataFrame:
    df = sql_fetch_df("SELECT * FROM visits")
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["checkin_ts"], errors="coerce").dt.date
    start, end = _period_bounds(periodo)
    if start:
        df = df[df["day"] >= start]
    if end:
        df = df[df["day"] <= end]
    if filial:
        df = df[df["filial"] == filial]
    if promotor_id:
        df = df[df["promotor_id"] == promotor_id]
    return df.copy()

def visits_today_df() -> pd.DataFrame:
    return visits_period_df("Hoje")

def promoters_from_users(filial: str | None = None) -> pd.DataFrame:
    q = "SELECT id, username, filial, promotor_id, promotor_nome, is_active FROM users WHERE role='promotor'"
    df = sql_fetch_df(q)
    if filial:
        df = df[df["filial"] == filial]
    return df

def promoter_status_table(periodo: str, filial: str | None) -> pd.DataFrame:
    users = promoters_from_users(filial)
    vt = visits_period_df(periodo, filial)
    if not vt.empty:
        agg = vt.groupby("promotor_id").agg(
            visitas=("id", "count"),
            concluidas=("checkout_ts", lambda s: s.notna().sum())
        ).reset_index()
    else:
        agg = pd.DataFrame(columns=["promotor_id", "visitas", "concluidas"])
    out = users.merge(agg, how="left", on="promotor_id")
    out["visitas"] = out["visitas"].fillna(0).astype(int)
    out["concluidas"] = out["concluidas"].fillna(0).astype(int)
    out["status_periodo"] = out["visitas"].apply(lambda x: "✅ realizou" if x > 0 else "⭕ não realizou")
    out = out.rename(columns={
        "username": "usuário",
        "promotor_nome": "promotor",
        "promotor_id": "id_promotor"
    })
    cols = ["usuário", "promotor", "filial", "id_promotor", "status_periodo", "visitas", "concluidas", "is_active"]
    return out[cols].sort_values(["filial", "promotor"])

# ========== MAPA ==========
def _done_today_idclientes() -> set:
    df = visits_today_df()
    if df.empty: return set()
    auth = st.session_state.get("auth") or {}
    if auth.get("role") == "promotor":
        df = df[(df["filial"] == auth.get("filial")) & (df["promotor_id"] == auth.get("promotor_id"))]
    return set(df["id_cliente"].astype(str).tolist())

def _build_route_map_df(force_promotor_id: str | None = None, force_schema: str | None = None) -> pd.DataFrame:
    df_map = st.session_state.get("df_routes_today", pd.DataFrame()).copy()
    if df_map.empty or (force_promotor_id is not None) or (force_schema is not None):
        filial = st.session_state.get("filial", "")
        schema = force_schema or st.session_state.get("schema") or (schema_from_filial(filial) if filial else "")
        if schema:
            engine_schema = get_engine_for_schema(schema)
            pid = force_promotor_id if force_promotor_id is not None else (st.session_state.get("promotor_id") or None)
            df_map = fetch_today_routes_with_client(engine_schema, pid)
            st.session_state.df_routes_today = df_map.copy()
    if df_map.empty:
        return df_map
    df_map["lat"] = pd.to_numeric(df_map.get("latitude"), errors="coerce")
    df_map["lon"] = pd.to_numeric(df_map.get("longitude"), errors="coerce")
    df_map = df_map.dropna(subset=["lat","lon"]).copy()
    if df_map.empty:
        return df_map
    done_ids = _done_today_idclientes()
    current_loja = str(st.session_state.get("loja_id") or "")
    has_checkin  = bool(st.session_state.get("active_visit_id"))
    def _is_done(row):
        cid = str(row.get("id_cliente"))
        if cid in done_ids: return True
        if has_checkin and cid == current_loja: return True
        return bool(row.get("realizado"))
    df_map["visitado_map"] = df_map.apply(_is_done, axis=1)
    df_map["color"] = df_map["visitado_map"].map(lambda v: [0,180,0] if v else [220,60,60])
    df_map["tooltip"] = df_map.apply(
        lambda r: f"{r.get('nome_cliente','(sem nome)')} • Cliente {r['id_cliente']} • "
                  f"{'Visitado' if r['visitado_map'] else 'Pendente'}",
        axis=1
    )
    return df_map

# ===================== TELAS =====================
def screen_home():
    require_auth()
    user = st.session_state.auth
    st.markdown("## 🏬 Amplificador de Demandas")

    if user["role"] == "promotor":
        filial = user["filial"]; promotor_id = user["promotor_id"]; promotor_nome = user["promotor_nome"]
        st.info(f"Filial: **{filial}** • Promotor: **{promotor_nome}** (id {promotor_id})")
        schema = schema_from_filial(filial); st.session_state.schema = schema
        engine_schema = get_engine_for_schema(schema)
        df_rot = fetch_today_routes_with_client(engine_schema, promotor_id)
        st.session_state.df_routes_today = df_rot.copy()
        if df_rot.empty:
            st.warning("Nenhuma loja para HOJE."); return
        opcoes = df_rot.apply(lambda r: f"{r['id_cliente']} — {r.get('nome_cliente','(sem nome)')}", axis=1).tolist()
        i = st.selectbox("Loja do roteiro (HOJE)", list(range(len(opcoes))), format_func=lambda k: opcoes[k])
        row = df_rot.iloc[i]
        st.session_state.filial = filial
        st.session_state.promotor_id = promotor_id
        st.session_state.promotor_nome = promotor_nome
        st.session_state.loja_id = str(row["id_cliente"])
        st.session_state.loja_nome = str(row.get("nome_cliente") or row.get("razao_social") or st.session_state.loja_id)
        if st.button("✅ Confirmar check-in", type="primary", use_container_width=True):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql_execute("""INSERT INTO visits (id_cliente, loja, filial, promotor_id, promotor_nome, checkin_ts)
                           VALUES (?,?,?,?,?,?)""",
                        (st.session_state.loja_id, st.session_state.loja_nome or st.session_state.loja_id,
                         filial, promotor_id, promotor_nome, ts))
            vid = sql_fetch_df("SELECT last_insert_rowid() AS id").iloc[0,0]
            st.session_state.active_visit_id = int(vid)
            st.success(f"Check-in: **{st.session_state.loja_nome or st.session_state.loja_id}**.")
            go("gondola")
    else:
        colA, colB, colC = st.columns([1.1, 1.6, 1.6])
        with colA:
            filial = st.selectbox("Filial", FILIAIS, index=FILIAIS.index(st.session_state.get("filial","TSJ")) if st.session_state.get("filial") in FILIAIS else 0)
            schema = schema_from_filial(filial)
            st.session_state.filial = filial; st.session_state.schema = schema
            st.caption(f"Schema MySQL: **{schema}**")
        engine_schema = get_engine_for_schema(schema)
        with colB:
            df_prom = fetch_promoters_today(engine_schema)
            if df_prom.empty:
                st.info("Sem promotores na `bi_roteiro` para HOJE.")
                st.session_state.promotor_id = ""; st.session_state.promotor_nome = ""
            else:
                opts = df_prom.apply(lambda r: f"{r['nome_promotor']} — {int(r['roteiros_cadastrados'])} roteiros", axis=1).tolist()
                idx = st.selectbox("Promotor (HOJE)", list(range(len(opts))), format_func=lambda i: opts[i])
                sel = df_prom.iloc[idx]
                st.session_state.promotor_id = str(sel["id_promotor"])
                st.session_state.promotor_nome = str(sel["nome_promotor"])
        with colC:
            df_rot = fetch_today_routes_with_client(engine_schema, st.session_state.promotor_id) if st.session_state.promotor_id else fetch_today_routes_with_client(engine_schema, None)
            st.session_state.df_routes_today = df_rot.copy()
            if df_rot.empty:
                st.info("Nenhuma loja no roteiro de HOJE.")
                st.session_state.loja_id = ""; st.session_state.loja_nome = ""
            else:
                opcoes = df_rot.apply(lambda r: f"{r['id_cliente']} — {r.get('nome_cliente','(sem nome)')}", axis=1).tolist()
                i = st.selectbox("Loja do roteiro (HOJE)", list(range(len(opcoes))), format_func=lambda k: opcoes[k])
                row = df_rot.iloc[i]
                st.session_state.loja_id = str(row["id_cliente"])
                st.session_state.loja_nome = str(row.get("nome_cliente") or row.get("razao_social") or st.session_state.loja_id)
        disabled = not (st.session_state.filial and st.session_state.loja_id)
        if st.button("✅ Confirmar check-in", type="primary", use_container_width=True, disabled=disabled):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql_execute("""INSERT INTO visits (id_cliente, loja, filial, promotor_id, promotor_nome, checkin_ts)
                           VALUES (?,?,?,?,?,?)""",
                        (st.session_state.loja_id, st.session_state.loja_nome or st.session_state.loja_id,
                         st.session_state.filial, st.session_state.promotor_id, st.session_state.promotor_nome, ts))
            vid = sql_fetch_df("SELECT last_insert_rowid() AS id").iloc[0,0]
            st.session_state.active_visit_id = int(vid)
            st.success(f"Check-in: **{st.session_state.loja_nome or st.session_state.loja_id}** — Filial **{st.session_state.filial}**.")
            go("gondola")

    st.markdown("### Etapas")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.button("🛒 Gôndola", on_click=lambda: go("gondola"), use_container_width=True, disabled=st.session_state.active_visit_id is None)
    with c2: st.button("📦 Estoque", on_click=lambda: go("estoque"), use_container_width=True, disabled=(st.session_state.active_visit_id is None or len(st.session_state.imgs_gondola)==0))
    with c3: st.button("✅ Checklist", on_click=lambda: go("guided"), use_container_width=True, disabled=st.session_state.active_visit_id is None)
    with c4: st.button("📊 Dashboard", on_click=lambda: go("dashboard"), use_container_width=True)

def screen_gondola():
    require_auth()
    st.header("🛒 Gôndola — fotos por câmera")
    with st.expander("🧾 Produtos da sua indústria (opcional)"):
        st.caption("Informe os seus produtos para a IA focar só neles. Cole um por linha e/ou envie um CSV/Excel (primeira coluna).")
        txt_cat = st.text_area("Lista (um por linha)", value="\n".join(st.session_state.get("catalogo_produtos", [])), height=140)
        up_cat = st.file_uploader("CSV/Excel (primeira coluna com os nomes)", type=["csv","xlsx","xls"], key="up_catalogo")
        if st.button("💾 Salvar catálogo para esta visita"):
            lista = parse_catalog_input(txt_cat, up_cat)
            st.session_state.catalogo_produtos = lista
            if st.session_state.active_visit_id:
                sql_execute("UPDATE visits SET catalogo_json=? WHERE id=?",
                            (json.dumps(lista, ensure_ascii=False), st.session_state.active_visit_id))
            st.success(f"{len(lista)} itens salvos para esta visita.")
    cap = st.camera_input("📸 Tirar foto da GÔNDOLA", key=f"cam_g_{st.session_state.cam_gondola_key}")
    if cap: add_captured_photo("imgs_gondola", cap)
    if st.button("➕ Outra foto"): st.session_state.cam_gondola_key += 1; st.rerun()
    if st.session_state.imgs_gondola:
        st.image([x["bytes"] for x in st.session_state.imgs_gondola], width=220)
    st.button("➡️ Ir para Estoque", type="primary", on_click=lambda: go("estoque"),
              disabled=len(st.session_state.imgs_gondola)==0, use_container_width=True)
    st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)

def screen_estoque():
    require_auth()
    st.header("📦 Estoque — fotos por câmera")
    if len(st.session_state.imgs_gondola)==0:
        st.warning("Antes, capture pelo menos 1 foto da GÔNDOLA."); return
    cap = st.camera_input("📸 Tirar foto do ESTOQUE", key=f"cam_e_{st.session_state.cam_estoque_key}")
    if cap: add_captured_photo("imgs_estoque", cap)
    if st.button("➕ Outra foto"): st.session_state.cam_estoque_key += 1; st.rerun()
    if st.session_state.imgs_estoque:
        st.image([x["bytes"] for x in st.session_state.imgs_estoque], width=220)
    st.button("✨ Gerar Checklist", type="primary", on_click=generate_combined_checklist,
              disabled=len(st.session_state.imgs_estoque)==0, use_container_width=True)
    st.button("⬅️ Voltar", on_click=lambda: go("gondola"), use_container_width=True)

# ===== IA com CATÁLOGO =====
def analyze_gondola(images: List[Image.Image]) -> str:
    cat = st.session_state.get("catalogo_produtos") or []
    cat_text = "\n".join(f"- {x}" for x in cat[:200])
    if cat:
        rules = (
            "CONSIDERE APENAS os produtos listados como 'nossos'. "
            "Compare por nome/variações/abreviações visíveis no rótulo e na etiqueta de preço. "
            "Se não corresponder claramente a um item da lista, classifique como 'NÃO É NOSSO'.\n\n"
            "Lista de produtos da nossa indústria:\n" + cat_text + "\n\n"
        )
    else:
        rules = ""
    prompt = (
        rules +
        "Você receberá 1..N fotos da GÔNDOLA. Responda em bullets curtos e objetivos:\n"
        "- % de VAZIO (baixo/médio/alto) e valor aproximado.\n"
        "- Participação por marca/SKU (facings) — relate apenas os nossos produtos se a lista foi fornecida.\n"
        "- Produtos identificáveis (nome/variação) e se o preço está visível; rotule outros itens como 'NÃO É NOSSO'.\n"
        "- Sugira QUANTIDADES para repor, blocagem e eye-level considerando apenas os nossos itens.\n"
        "Se houver dúvida de correspondência, escreva 'incerto' e peça confirmação.\n"
    )
    return gpt_vision_multi(prompt, images, max_tokens=1000)

def analyze_estoque(images: List[Image.Image]) -> str:
    cat = st.session_state.get("catalogo_produtos") or []
    cat_text = "\n".join(f"- {x}" for x in cat[:200])
    if cat:
        rules = (
            "CONSIDERE APENAS os produtos listados como 'nossos' ao contar e projetar estoque. "
            "Se não corresponder claramente a um item da lista, classifique como 'NÃO É NOSSO'.\n\n"
            "Lista de produtos da nossa indústria:\n" + cat_text + "\n\n"
        )
    else:
        rules = ""
    prompt = (
        rules +
        "Você receberá 1..N fotos do ESTOQUE. Responda em bullets curtos:\n"
        "- Quantidade por SKU (aprox) e confiança (apenas nossos itens, se lista fornecida).\n"
        "- Consumo médio/dia; se confiança baixa, sinalize 'pedir input manual do vendedor'.\n"
        "- Projeção de queda e possíveis rupturas.\n"
        "- Recomendações de reposição alinhadas à gôndola (sequência, facing, eye-level, blocagem, preço).\n"
    )
    return gpt_vision_multi(prompt, images, max_tokens=1000)

def checklist_from_analysis(txt: str) -> List[str]:
    pats=[r"^claro",r"^aqui está",r"^segue",r"^resumo",r"^checklist"]
    lines=[l.strip() for l in (txt or "").splitlines() if l.strip()]
    i=0
    while i<len(lines) and (lines[i].startswith("#") or any(re.search(p, lines[i].lower()) for p in pats)): i+=1
    body="\n".join(lines[i:]) if i<len(lines) else (txt or "")
    steps=[]
    for raw in body.splitlines():
        s=re.sub(r'^\s*[-•*\d\.\)]\s*','',raw.strip())
        s=re.sub(r'^\s{0,3}#{1,6}\s*','',s)
        if s: steps.append(re.sub(r'\s+',' ',s)[:180])
    if not steps:
        steps=[p.strip() for p in re.split(r"[\.!?]\s+", body) if p.strip()]
    return steps[:20] or ["Refaça as fotos com melhor iluminação."]

def generate_combined_checklist():
    require_auth()
    imgs_g = photos_to_pil("imgs_gondola"); imgs_e = photos_to_pil("imgs_estoque")
    with st.spinner("IA analisando gôndola..."): ag = analyze_gondola(imgs_g) if imgs_g else "Sem fotos da gôndola."
    with st.spinner("IA analisando estoque..."): ae = analyze_estoque(imgs_e) if imgs_e else "Sem fotos do estoque."
    st.session_state.analysis_gondola, st.session_state.analysis_estoque = ag, ae
    cat = st.session_state.get("catalogo_produtos") or []
    cat_text = "\n".join(f"- {x}" for x in cat[:200])
    cat_block = f"**Considere apenas estes produtos como 'nossos':**\n{cat_text}\n\n" if cat else ""
    client = ensure_openai_client()
    if client:
        try:
            content = (cat_block + "RESUMO GÔNDOLA:\n" + ag + "\n\n" + "RESUMO ESTOQUE:\n" + ae + "\n\n" +
                       "Gere um checklist final (passos curtos e acionáveis, na ordem de execução) "
                       "apenas para os nossos itens (se fornecidos), cobrindo reposição, facing, eye-level, blocagem e preço visível.")
            r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, max_tokens=900,
                                               messages=[{"role":"system","content":"Você é um consultor de execução em PDV."},
                                                         {"role":"user","content":content}])
            txt = (r.choices[0].message.content or "").strip()
        except Exception as e:
            txt = f"(Falha ao gerar checklist com IA: {e})"
    else:
        txt = "IA desativada — defina OPENAI_API_KEY."
    st.session_state.guided = {"steps": checklist_from_analysis(txt), "idx": 0}
    vid = st.session_state.active_visit_id
    if vid:
        sql_execute("UPDATE visits SET analysis_gondola=?, analysis_estoque=? WHERE id=?",(ag, ae, vid))
    go("guided")

def screen_guided():
    require_auth()
    g = st.session_state.guided
    steps: List[str] = g.get("steps", [])
    idx: int = g.get("idx", 0)
    if not steps:
        st.info("Gere o checklist depois de capturar fotos de gôndola e estoque.")
        st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True); return
    total = len(steps); atual = steps[idx]
    st.markdown("### ✅ Checklist guiado (um passo por vez)")
    st.progress(idx/total if total else 0, text=f"Etapa {idx+1} de {total}")
    st.markdown(f"**Passo atual:** {atual}")
    if st.button("🔊 Ouvir instrução", use_container_width=True):
        audio_bytes, mime = tts_bytes(atual)
        if audio_bytes: st.audio(audio_bytes, format=mime)
    st.markdown("Confirme (ex.: 'ok', 'feito') ou registre observação:")
    saida_txt = st.text_input("", key=f"txt_{idx}"); ok_text = st.button("✅ Confirmar por texto", use_container_width=True)
    saida_voice = ""; ok_voice_clicked = False
    if MIC_OK:
        st.caption("Ou toque no microfone e diga **'ok'** / **'feito'**.")
        audio = mic_recorder(start_prompt="🎙️ Gravar", stop_prompt="Parar", just_once=True, format="webm", key=f"mic_{idx}")
        ok_voice_clicked = st.button("✅ Confirmar por voz", use_container_width=True)
        if ok_voice_clicked and audio and isinstance(audio, dict) and audio.get("bytes"):
            saida_voice = stt_from_bytes(audio["bytes"])
            if saida_voice: st.caption(f"Transcrição: _{saida_voice}_")
    else:
        st.caption("🎙️ Para voz, instale `streamlit-mic-recorder`.")
    avancar = False
    if ok_text and saida_txt and any(k in saida_txt.lower() for k in ["ok","feito","concluído","concluido"]): avancar = True
    if MIC_OK and ok_voice_clicked and saida_voice and any(k in saida_voice for k in ["ok","feito","concluído","concluido"]): avancar = True
    colX, colY, colZ = st.columns(3)
    with colX:
        if st.button("⏭️ Pular passo (não recomendado)", use_container_width=True): avancar = True
    with colY:
        st.button("❌ Cancelar", on_click=lambda: go("home"), use_container_width=True)
    with colZ:
        st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)
    if avancar:
        novo = idx + 1
        if novo >= total: go("audit_final"); st.rerun()
        else: st.session_state.guided["idx"] = novo; st.rerun()

def _parse_level(txt: str) -> str:
    t=(txt or "").upper()
    if "EXCELENTE" in t: return "EXCELENTE"
    if "MELHORAR" in t or "RUIM" in t: return "MELHORAR"
    return "OK"

def analyze_final_gondola(before: List[Image.Image], after: Image.Image):
    client = ensure_openai_client()
    if not client: return "IA desligada.", "OK"
    content=[{"type":"text","text":
              "Compare fotos ANTES (1..N) com a DEPOIS. Diga 'NÍVEL: <EXCELENTE|OK|MELHORAR>' e 3–6 sugestões objetivas. Curto e direto."}]
    for img in before: content.append({"type":"image_url","image_url":{"url":image_to_b64_url(img)}})
    content.append({"type":"text","text":"--- Foto DEPOIS ---"})
    content.append({"type":"image_url","image_url":{"url":image_to_b64_url(after)}})
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, max_tokens=650,
                                           messages=[{"role":"system","content":"Auditor PDV objetivo."},
                                                     {"role":"user","content":content}])
        txt=(r.choices[0].message.content or "").strip()
    except Exception as e:
        txt=f"Falha IA: {e}"
    return txt, _parse_level(txt)

def screen_audit_final():
    require_auth()
    st.header("🏁 Revisão final — Gôndola (antes x depois)")
    if not st.session_state.imgs_gondola:
        st.info("Faça pelo menos 1 foto da gôndola antes."); return
    cap = st.camera_input("📸 Foto FINAL da gôndola", key=f"cam_f_{st.session_state.cam_final_key}")
    if cap: st.session_state.final_gondola = cap.getvalue()
    if st.button("🔁 Nova final"): st.session_state.cam_final_key += 1; st.rerun()
    can = bool(st.session_state.final_gondola)
    if st.button("🔎 Analisar e comparar", type="primary", disabled=not can, use_container_width=True):
        before = photos_to_pil("imgs_gondola"); after = Image.open(io.BytesIO(st.session_state.final_gondola)).convert("RGB")
        txt, level = analyze_final_gondola(before, after)
        st.session_state.final_review_text, st.session_state.final_review_level = txt, level
        try:
            vaz_a, vaz_d = estimate_empty_before_after(before[0], after)
            st.session_state.empty_before, st.session_state.empty_after = vaz_a, vaz_d
            st.session_state.empty_delta = (vaz_d - vaz_a) if vaz_a is not None and vaz_d is not None else None
        except Exception:
            st.session_state.empty_before = st.session_state.empty_after = st.session_state.empty_delta = None
        st.rerun()
    if st.session_state.final_review_text:
        badge={"EXCELENTE":"🟢","OK":"🟡","MELHORAR":"🟠"}.get(st.session_state.final_review_level,"🟡")
        st.write(f"**Nível:** {badge} **{st.session_state.final_review_level}**")
        st.text_area("Análise da IA", st.session_state.final_review_text, height=220)
        col1,col2,col3=st.columns(3)
        with col1: st.metric("Vazio antes (~%)", st.session_state.empty_before if st.session_state.empty_before is not None else "—")
        with col2: st.metric("Vazio depois (~%)", st.session_state.empty_after if st.session_state.empty_after is not None else "—")
        with col3:
            delta = st.session_state.empty_delta
            st.metric("Variação (pp)", f"{delta:+d}" if isinstance(delta,int) else "—")
    def finalize():
        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vid=st.session_state.active_visit_id
        if not vid: return
        dfc=sql_fetch_df("SELECT checkin_ts FROM visits WHERE id=?", (vid,))
        dur=minutes_between(dfc.loc[0,"checkin_ts"], now) if not dfc.empty else None
        result={"guided": st.session_state.guided.get("steps", []),
                "final_review":{"level":st.session_state.final_review_level,
                                "text":st.session_state.final_review_text,
                                "empty_before":st.session_state.empty_before,
                                "empty_after":st.session_state.empty_after,
                                "empty_delta":st.session_state.empty_delta}}
        sql_execute("""UPDATE visits
                          SET result_json=?, checklist_json=?, checkout_ts=?, duracao_min=?, eficiencia=?
                        WHERE id=?""",
                    (json.dumps(result,ensure_ascii=False),
                     json.dumps(st.session_state.guided.get("steps", []), ensure_ascii=False),
                     now, dur, 1.0, vid))
        st.session_state.active_visit_id=None
    c1,c2=st.columns(2)
    with c1:
        if st.button("✅ Concluir visita", type="primary", use_container_width=True):
            finalize(); st.success("Visita finalizada."); go("dashboard")
    with c2:
        st.button("⬅️ Voltar", on_click=lambda: go("guided"), use_container_width=True)

# ===================== Dashboard (com seletor de período) =====================
def screen_dashboard():
    require_auth()
    auth = st.session_state.auth
    st.header("📊 Dashboard")

    # ----- seletor de período -----
    with st.container():
        periodo = st.segmented_control("Período", options=["Hoje","Semana","Mês","Tudo"], default="Hoje")

    # ============================ PROMOTOR ============================
    if auth["role"] == "promotor":
        vt = visits_period_df(periodo, filial=auth["filial"], promotor_id=auth["promotor_id"])
        total = len(vt)
        concl = int(vt["checkout_ts"].notna().sum()) if not vt.empty else 0
        ef = round((concl/total)*100, 1) if total else 0.0
        tmedio = round(vt["duracao_min"].dropna().mean(), 2) if "duracao_min" in vt.columns and not vt["duracao_min"].dropna().empty else 0
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Visitas", total)
        with c2: st.metric("Concluídas", concl)
        with c3: st.metric("Tempo médio (min)", tmedio)
        with c4: st.metric("Eficiência", f"{ef}%")

        st.markdown("#### 🗺️ Mapa — Roteiro de HOJE")
        dfm = _build_route_map_df()
        if dfm.empty:
            st.info("Sem visitas programadas para HOJE ou sem coordenadas.")
        else:
            import pydeck as pdk
            layer = pdk.Layer("ScatterplotLayer", data=dfm, get_position='[lon, lat]', get_fill_color='color', get_radius=70, pickable=True)
            view = pdk.ViewState(latitude=float(dfm["lat"].mean()), longitude=float(dfm["lon"].mean()), zoom=10)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{tooltip}"}))
            cols = ["id_cliente","nome_cliente","data_visita","status","visitado_map","latitude","longitude"]
            cols = [c for c in cols if c in dfm.columns]
            st.dataframe(dfm[cols].rename(columns={"visitado_map":"visitado"}), use_container_width=True, hide_index=True)
        st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)
        return

    # ============================ ADMIN ============================
    fil_col, pro_col = st.columns([1,2])
    with fil_col:
        filial_opt = st.selectbox("Filtrar por filial", ["Todas"] + FILIAIS, index=0)
        filial_sel = None if filial_opt == "Todas" else filial_opt
    with pro_col:
        df_proms = promoters_from_users(filial_sel)
        pro_labels = ["Todos"] + [f"{r['promotor_nome']} (id {r['promotor_id']}) — {r['filial']}" for _, r in df_proms.iterrows()]
        idx = st.selectbox("Filtrar por promotor", list(range(len(pro_labels))), format_func=lambda i: pro_labels[i])
        prom_sel_id = None if idx == 0 else str(df_proms.iloc[idx-1]["promotor_id"])
        prom_sel_filial = None if idx == 0 else str(df_proms.iloc[idx-1]["filial"])

    vt = visits_period_df(periodo, filial=filial_sel, promotor_id=prom_sel_id)
    total = len(vt)
    concl = int(vt["checkout_ts"].notna().sum()) if not vt.empty else 0
    ef = round((concl/total)*100, 1) if total else 0.0
    tmedio = round(vt["duracao_min"].dropna().mean(), 2) if "duracao_min" in vt.columns and not vt["duracao_min"].dropna().empty else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Visitas", total)
    with c2: st.metric("Concluídas", concl)
    with c3: st.metric("Tempo médio (min)", tmedio)
    with c4: st.metric("Eficiência", f"{ef}%")

    st.markdown(f"#### 👥 Status dos promotores ({periodo})")
    df_status = promoter_status_table(periodo, filial_sel)
    if prom_sel_id:
        df_status = df_status[df_status["id_promotor"].astype(str) == prom_sel_id]
    if df_status.empty:
        st.info("Nenhum promotor cadastrado para este filtro.")
    else:
        st.dataframe(df_status, use_container_width=True, hide_index=True)

    st.markdown("#### 🗺️ Mapa — Roteiro de HOJE (por filtro)")
    force_schema = schema_from_filial(prom_sel_filial) if prom_sel_filial else (schema_from_filial(filial_sel) if filial_sel else None)
    dfm = _build_route_map_df(force_promotor_id=prom_sel_id, force_schema=force_schema)
    if dfm.empty:
        st.info("Sem visitas programadas para HOJE (ou selecione uma filial/promotor com roteiro).")
    else:
        import pydeck as pdk
        layer = pdk.Layer("ScatterplotLayer", data=dfm, get_position='[lon, lat]', get_fill_color='color', get_radius=70, pickable=True)
        view = pdk.ViewState(latitude=float(dfm["lat"].mean()), longitude=float(dfm["lon"].mean()), zoom=10)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{tooltip}"}))
        st.markdown("#### 📄 Clientes do roteiro (HOJE)")
        cols = ["id_cliente","nome_cliente","data_visita","status","visitado_map","latitude","longitude"]
        cols = [c for c in cols if c in dfm.columns]
        st.dataframe(dfm[cols].rename(columns={"visitado_map":"visitado"}), use_container_width=True, hide_index=True)

    st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)

# ===== Sidebar =====
with st.sidebar:
    if st.session_state.auth:
        st.markdown(f"**Usuário:** {st.session_state.auth['username']}  \n**Perfil:** {st.session_state.auth['role']}")
        st.button("🏠 Início", on_click=lambda: go("home"), use_container_width=True)
        st.button("🆕 Nova loja", on_click=lambda: st.session_state.update({
            "loja_id":"", "loja_nome":"", "active_visit_id":None,
            "imgs_gondola":[], "imgs_estoque":[], "guided":{"steps":[], "idx":0},
            "final_gondola":None, "final_review_text":"", "final_review_level":"",
            "empty_before":None, "empty_after":None, "empty_delta":None,
            "cam_gondola_key":0, "cam_estoque_key":0, "cam_final_key":0,
            "catalogo_produtos":[]
        }) or go("home"), use_container_width=True)
        st.button("📊 Dashboard", on_click=lambda: go("dashboard"), use_container_width=True)
        if st.session_state.auth["role"]=="admin":
            st.button("🛠️ Admin", on_click=lambda: go("admin"), use_container_width=True)
        if st.session_state.active_visit_id:
            if st.button("🔚 Forçar check-out", use_container_width=True):
                now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dfc=sql_fetch_df("SELECT checkin_ts FROM visits WHERE id=?", (st.session_state.active_visit_id,))
                dur=minutes_between(dfc.loc[0,"checkin_ts"], now) if not dfc.empty else None
                sql_execute("UPDATE visits SET checkout_ts=?, duracao_min=? WHERE id=?",
                            (now, dur, st.session_state.active_visit_id))
                st.success("Check-out concluído.")
                st.session_state.active_visit_id=None
        if st.button("🚪 Sair", use_container_width=True):
            st.session_state.auth=None; go("login"); st.rerun()
    else:
        st.markdown("### Bem-vindo")
        st.caption("Faça login para continuar.")

# ===== Roteador =====
screen = st.session_state.screen
if screen == "login" and st.session_state.get("auth"):
    go("home"); st.rerun()
if screen == "login":        screen_login()
elif screen == "home":       screen_home()
elif screen == "gondola":    screen_gondola()
elif screen == "estoque":    screen_estoque()
elif screen == "guided":     screen_guided()
elif screen == "audit_final":screen_audit_final()
elif screen == "dashboard":  screen_dashboard()
elif screen == "admin":      screen_admin()
else:                        screen_login()
