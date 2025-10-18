# lastmile_app.py — Last Mile - Loja Perfeita (Login + Admin + MySQL + IA + PDF + Mapa + Relatórios + Kanban de Pedidos + Consulta + Integração ERP + Pedido Manual)
# ---------------------------------------------------------------------------------------------------
# Versão com integração para enviar pedidos para a API "Clube da Venda" e criação de pedidos manuais a partir de uma busca no banco de dados.
# Dependências adicionais: requests (pip install requests pydantic)

import os, io, re, json, base64, sqlite3, tempfile, datetime, hashlib, secrets, uuid, requests
from typing import Optional, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field

# ==== .env ====
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ==== Streamlit ====
st.set_page_config(page_title="Last Mile — Loja Perfeita", page_icon="🛒", layout="wide")

# ==== Configuração da API Externa (Clube da Venda) ====
CLUBE_VENDA_API_URL = os.getenv("CLUBE_VENDA_API_URL", "https://sales.tradepro.com.br/servicos/api/v1/inserir-pedido")
# Dicionário de tokens por filial
CLUBE_VENDA_API_TOKENS = {
    "TPH": "eq2znwn5QBDv",
    # Adicionar tokens para outras filiais aqui. Ex: "ABC": "SEU_TOKEN_DA_ABC"
    "DEFAULT": "" # Token padrão ou vazio se não houver
}

# ==== Esquemas Pydantic para Validação do Pedido ====
class ProdutoRef(BaseModel):
    id: str = Field(..., description="ID do produto no ERP")

class ItemPedidoSchema(BaseModel):
    percentualDm: float = 0
    precoUnitario: float = Field(..., ge=0)
    quantidade: int = Field(..., ge=1)
    totalDesconto: float = 0
    totalItem: float = Field(..., ge=0)
    valorTrocaNaoAutorizada: float = 0
    produto: ProdutoRef

class ClienteRef(BaseModel):
    id: str = Field(..., description="ID (CNPJ/CPF ou ID interno) do cliente no ERP")

class PedidoSchema(BaseModel):
    dataPedido: str = Field(..., description="YYYY-MM-DD")
    prazoMedio: int = 0
    totalDesconto: float = 0
    custoPedido: float = 0
    saldoDm: float = 0
    margem: float = 0
    totalPedido: float = Field(..., ge=0)
    sincronizado: int = 0
    cliente: ClienteRef
    notificarTroca: int = 0
    trocaEnviada: int = 0
    enviarEmail: int = 0
    emailEnviado: int = 0
    confirmado: int = 0
    ecommerce: int = 0
    origem: str = "ECOMMERCE"
    itensPedido: List[ItemPedidoSchema]

def validate_payload(payload: dict) -> Optional[str]:
    """Valida o payload do pedido usando Pydantic e regras de negócio."""
    try:
        PedidoSchema(**payload)
        soma = round(sum(i["totalItem"] for i in payload.get("itensPedido", [])), 2)
        if abs(soma - float(payload.get("totalPedido", 0))) > 0.01:
            return f"totalPedido ({payload.get('totalPedido')}) difere da soma dos itens ({soma})."
        datetime.datetime.strptime(payload["dataPedido"], "%Y-%m-%d")
        return None
    except Exception as e:
        return str(e)


# ==== MySQL (SQLAlchemy + PyMySQL) ====
from urllib.parse import quote_plus, quote
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

@st.cache_resource
def get_engine_for_schema(schema: str):
    uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/{schema}?charset=utf8mb4"
    return create_engine(uri, pool_pre_ping=True)

FILIAIS = ["ABC","API","MCD","TBE","TBL","TCA","TCG","TCV","TPA","TPH","TSJ"]
FILIAL_TO_SCHEMA = {
    "ABC":"promoter_abc","API":"promoter_api2","MCD":"mcd",
    "TBE":"promoter_tbe","TBL":"promoter_tbl","TCA":"promoter_tca",
    "TCG":"mtcg","TCV":"promoter_tcv","TPA":"promoter_tpa",
    "TPH":"promoter_tph2","TSJ":"promoter_tsj"
}
def schema_from_filial(filial: str) -> str:
    return FILIAL_TO_SCHEMA.get(filial.upper(), f"promoter_{filial.lower()}")

# ==== Roteiro: mapear colunas tolerante ====
def _map_roteiro_columns(conn, schema: str) -> dict:
    try:
        rows = conn.execute(text(f"SHOW COLUMNS FROM `{schema}`.`roteiro`")).fetchall()
    except Exception:
        return {"id":"id","cliente":"id_cliente","rep":"id_promotor","data":"data_visita","status":"status","real":"realizado"}
    cols = {r[0].lower(): r[0] for r in rows}
    f = lambda cands, d=None: next((cols[c] for c in cands if c in cols), d)
    return {
        "id":     f(["id","id_roteiro"],"id"),
        "cliente":f(["id_cliente","cliente_id","id_cliente_fk","cliente"],"id_cliente"),
        "rep":    f(["id_promotor","id_representante","id_vendedor","id_usuario"], None),
        "data":   f(["data_visita","data","dt_visita"],"data_visita"),
        "status": f(["status","situacao"],"status"),
        "real":   f(["realizado","dt_realizado","data_realizacao"], None),
    }

@st.cache_data(ttl=3600) # Cache por 1 hora
def fetch_products(_engine_schema) -> pd.DataFrame:
    try:
        with _engine_schema.connect() as conn:
            df = pd.read_sql(text("SELECT codigo, descricao FROM produto WHERE ativo=1 ORDER BY descricao"), conn)
            df['codigo'] = df['codigo'].astype(str)
            df['descricao'] = df['descricao'].astype(str)
            return df
    except Exception as e:
        st.error(f"Não foi possível carregar a lista de produtos: {e}")
        return pd.DataFrame(columns=['codigo', 'descricao'])

def fetch_promoters_today(engine_schema) -> pd.DataFrame:
    q = text("""
        SELECT DATE(`data`) AS data,
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
            r.`{m['id']}`      AS id,
            r.`{m['cliente']}`   AS id_cliente,
            {rep_sel}            AS id_promotor,
            r.`{m['data']}`      AS data_visita,
            r.`{m['status']}`    AS status,
            {real_sel}           AS realizado,
            c.fantasia           AS nome_cliente,
            c.razao_social,
            c.latitude,
            c.longitude
        FROM `roteiro` r
        LEFT JOIN `cliente` c ON c.id = r.`{m['cliente']}`
        WHERE DATE(r.`{m['data']}`) = CURDATE()
        """
        params = {}
        if id_promotor and m["rep"]:
            base_sql += f" AND r.`{m['rep']}` = :pid"
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

@st.cache_data(ttl=3600) # Cache por 1 hora
def fetch_all_clients_for_promoter(_engine_schema, promotor_id: str) -> pd.DataFrame:
    if not promotor_id:
        return pd.DataFrame()
    try:
        with _engine_schema.connect() as conn:
            m = _map_roteiro_columns(conn, _engine_schema.url.database)
            if not m.get("rep"): 
                return pd.DataFrame()

            client_ids_query = f"""
                SELECT DISTINCT r.`{m['cliente']}` as id_cliente
                FROM `roteiro` r
                WHERE r.`{m['rep']}` = :pid
            """
            
            clients_query = f"""
                SELECT c.id, c.fantasia as nome_cliente, c.razao_social
                FROM `cliente` c
                WHERE c.id IN ({client_ids_query})
                ORDER BY c.fantasia
            """
            
            params = {"pid": promotor_id}
            df = pd.read_sql(text(clients_query), conn, params=params)
            
            if not df.empty:
                df["nome_cliente"] = df.get("nome_cliente").fillna(df.get("razao_social"))
                df['id'] = df['id'].astype(str)
            return df
    except Exception as e:
        st.warning(f"Erro ao buscar clientes do promotor: {e}")
        return pd.DataFrame()

# ==== OpenAI (IA) ====
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
    content = [{"type":"text","text":prompt}] + [{"type":"image_url","image_url":{"url":image_to_b64_url(i)}} for i in images]
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.3, max_tokens=max_tokens,
            messages=[{"role":"system","content":"Você é um consultor de execução em PDV, direto e prático."},
                      {"role":"user","content":content}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Falha IA: {e}"

# === Mic opcional
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_OK = True
except Exception:
    MIC_OK = False

# === TTS (sanitiza símbolos p/ voz)
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
    except Exception:
        try:
            with client.audio.speech.with_streaming_response.create(model="gpt-4o-mini-tts", voice="alloy", input=texto) as stream:
                return stream.read(), "audio/mp3"
        except Exception as e:
            st.warning(f"TTS falhou: {e}")
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

# === % vazio
def _estimate_empty_percent(img: Image.Image):
    client = ensure_openai_client()
    if not client: return None
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=10,
            messages=[{"role":"user","content":[
                {"type":"text","text":"Responda apenas um número 0..100: % de área VAZIA na gôndola."},
                {"type":"image_url","image_url":{"url": image_to_b64_url(img)}}
            ]}])
        m = re.search(r"(\d{1,3})", (r.choices[0].message.content or ""))
        if not m: return None
        return max(0, min(100, int(m.group(1))))
    except Exception:
        return None

def estimate_empty_before_after(before_img: Image.Image, after_img: Image.Image):
    return _estimate_empty_percent(before_img), _estimate_empty_percent(after_img)

# ==== SQLite (visits + users + orders) ====
DATA_DIR = os.getenv("LASTMILE_DATA_DIR") or os.getenv("RENDER_DISK_PATH") or os.path.dirname(__file__)
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.getenv("LASTMILE_DB_PATH") or os.path.join(DATA_DIR, "lastmile.db")

def sql_execute(q, params=()):
    con = sqlite3.connect(DB_PATH); cur = con.cursor(); cur.execute(q, params); con.commit(); con.close()
def sql_fetch_df(q, params=()):
    con = sqlite3.connect(DB_PATH); df = pd.read_sql_query(q, con, params=params); con.close(); return df

# password hash
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
    sql_execute("""CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_cliente TEXT,
        loja TEXT,
        filial TEXT,
        promotor_id TEXT,
        promotor_nome TEXT,
        lat REAL,
        lon REAL,
        checkin_ts TEXT,
        checkout_ts TEXT,
        duracao_min REAL,
        eficiencia REAL,
        analysis_gondola TEXT,
        analysis_estoque TEXT,
        checklist_json TEXT,
        result_json TEXT,
        catalogo_json TEXT
    )""")
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("PRAGMA table_info(visits)")
    have = {r[1] for r in cur.fetchall()}
    def need(col, ddl):
        if col not in have: cur.execute(ddl)
    need("purchase_pdf_path", "ALTER TABLE visits ADD COLUMN purchase_pdf_path TEXT")
    con.commit()

    # users
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        pass_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        filial TEXT,
        promotor_id TEXT,
        promotor_nome TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TEXT
    )""")
    con.commit()

    # orders
    cur.execute("""CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id INTEGER,
        id_cliente TEXT,
        loja TEXT,
        filial TEXT,
        promotor_id TEXT,
        promotor_nome TEXT,
        created_ts TEXT,
        items_json TEXT,
        status TEXT NOT NULL,
        FOREIGN KEY (visit_id) REFERENCES visits (id)
    )""")
    con.commit()

    # seed admin
    df_admin = pd.read_sql_query("SELECT id FROM users WHERE username='admin'", con)
    if df_admin.empty:
        admin_pw = os.getenv("ADMIN_PASSWORD", "admin123")
        cur.execute("INSERT INTO users (username, pass_hash, role, is_active, created_at) VALUES (?,?,?,?,?)",
                    ("admin", _hash_pw(admin_pw), "admin", 1, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        con.commit()
    con.close()
init_db()

# users helpers
def user_get(username: str) -> Optional[dict]:
    df = sql_fetch_df("SELECT * FROM users WHERE username=?", (username,))
    return df.to_dict("records")[0] if not df.empty else None
def user_list() -> pd.DataFrame:
    return sql_fetch_df("SELECT id, username, role, filial, promotor_id, promotor_nome, is_active, created_at FROM users ORDER BY username")
def user_create(username: str, password: str, role: str, filial: Optional[str], promotor_id: Optional[str], promotor_nome: Optional[str]):
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

# ==== Sessão / utilitários ====
def ss_init():
    preset = {
        "screen":"login",
        "auth": None,
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
        "catalogo_produtos":[],
        "no_estoque": False,
        "visit_just_finalized": False,
        "manual_order_items": [],
        "audio_to_play": None # <-- ATUALIZADO
    }
    for k,v in preset.items(): st.session_state.setdefault(k,v)
ss_init()
def go(x): st.session_state.screen = x
def require_auth():
    if not st.session_state.auth:
        go("login"); st.stop()

def seg_ctrl(label, options, default):
    try:
        return st.segmented_control(label, options=options, default=default)
    except Exception:
        return st.radio(label, options, index=options.index(default), horizontal=True)

# fotos/tempo
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

# catálogo (textarea + CSV/Excel)
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

# ==== IA (análises)
def analyze_gondola(images: List[Image.Image]) -> str:
    cat = st.session_state.get("catalogo_produtos") or []
    cat_text = "\n".join(f"- {x}" for x in cat[:200])
    rules = ""
    if cat:
        rules = ("CONSIDERE APENAS os produtos listados como 'nossos'. Compare por nome/variações/abreviações "
                 "visíveis no rótulo e na etiqueta de preço. Se não corresponder claramente, rotule 'NÃO É NOSSO'.\n\n"
                 "Lista de produtos:\n" + cat_text + "\n\n")
    prompt = (rules +
        "Você receberá 1..N fotos da GÔNDOLA. Responda em bullets curtos:\n"
        "- % de VAZIO (baixo/médio/alto) e valor aproximado.\n"
        "- Participação por marca/SKU (facings) — relate apenas os nossos itens, se a lista foi fornecida.\n"
        "- Produtos identificáveis (nome/variação) e se o preço está visível; rotule outros itens como 'NÃO É NOSSO'.\n"
        "- Sugira QUANTIDADES para repor, blocagem e eye-level, focando nos nossos itens.\n"
        "Se houver dúvida de correspondência, escreva 'incerto' e peça confirmação.\n")
    return gpt_vision_multi(prompt, images, max_tokens=1000)

def analyze_estoque(images: List[Image.Image]) -> str:
    cat = st.session_state.get("catalogo_produtos") or []
    cat_text = "\n".join(f"- {x}" for x in cat[:200])
    rules = ""
    if cat:
        rules = ("CONSIDERE APENAS os produtos listados como 'nossos' ao contar/projetar estoque. "
                 "Se não corresponder claramente, rotule 'NÃO É NOSSO'.\n\n"
                 "Lista de produtos:\n" + cat_text + "\n\n")
    prompt = (rules +
        "Você receberá 1..N fotos do ESTOQUE. Responda em bullets curtos:\n"
        "- Quantidade por SKU (aprox) e confiança (apenas nossos itens, se lista fornecida).\n"
        "- Consumo médio/dia; se confiança baixa, sinalize 'pedir input manual do vendedor'.\n"
        "- Projeção de queda e possíveis rupturas.\n"
        "- Recomendações de reposição alinhadas à gôndola (sequência, facing, eye-level, blocagem, preço).\n")
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

# ==== PDF
def _save_purchase_pdf(loja: str, filial: str, promotor: str, sugestao_texto: str) -> str:
    out_dir = os.path.join(DATA_DIR, "compras_pdf")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    loja = loja or ""
    safe_loja = re.sub(r"[^\w]+", "_", loja, flags=re.UNICODE)[:32]
    fname = f"compra_{filial}_{safe_loja}_{ts}.pdf"
    path = os.path.join(out_dir, fname)
    content = (
        f"Relatório de Compra\nLoja: {loja}\nFilial: {filial}\nPromotor: {promotor}\n"
        f"Data: {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n{sugestao_texto}"
    )
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4
        text_obj = c.beginText(2*cm, h - 2*cm)
        text_obj.setFont("Helvetica", 10)
        for line in content.splitlines():
            text_obj.textLine(line)
        c.drawText(text_obj)
        c.save()
        return path
    except Exception:
        try:
            from fpdf import FPDF
            pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in content.splitlines():
                pdf.multi_cell(0, 6, line)
            pdf.output(path)
            return path
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return path

# ==== Sugestão de compra (IA + regras)
def _build_purchase_suggestions() -> dict:
    client = ensure_openai_client()
    gond = st.session_state.get("analysis_gondola", "") or ""
    est  = st.session_state.get("analysis_estoque", "") or ""
    rev  = st.session_state.get("final_review_text", "") or ""
    empty_after = st.session_state.get("empty_after", None)
    no_estoque = bool(st.session_state.get("no_estoque", False))
    cat = st.session_state.get("catalogo_produtos") or []
    texto = (gond + " " + est + " " + rev).lower()
    achou_ruptura = any(k in texto for k in ["ruptura","rupturas","sem estoque","sem-estoque","vazio","esgotado"])
    vazio_alto = isinstance(empty_after, (int,float)) and empty_after >= 25
    precisa_compra = no_estoque or achou_ruptura or vazio_alto
    itens=[]
    if client:
        prompt = (
            "Você é um planejador de compras. Gere JSON: {\"itens\":[{\"produto\":\"...\",\"quantidade\":<int>,\"justificativa\":\"...\"}, ...]} "
            "Foque só nos 'nossos' produtos se lista for fornecida. Se nada claro, retorne {\"itens\":[]}\n\n"
            f"CATÁLOGO: {', '.join(cat[:100])}\n\n"
            "ANÁLISE GÔNDOLA:\n" + gond + "\n\n" +
            "ANÁLISE ESTOQUE:\n" + est + "\n\n" +
            "REVISÃO FINAL:\n" + rev + f"\n\nVazio final: {empty_after}% | Loja sem estoque físico? {no_estoque}\n"
        )
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.1, max_tokens=500,
                messages=[{"role":"system","content":"Responda apenas JSON válido."},
                          {"role":"user","content":prompt}]
            )
            raw = (r.choices[0].message.content or "").strip()
            raw = re.sub(r"^```json|```$", "", raw).strip()
            data = json.loads(raw) if raw else {"itens":[]}
            itens = data.get("itens", [])
        except Exception:
            itens = []
    else:
        if precisa_compra:
            itens = [{"produto":"Itens de alto giro","quantidade":6,"justificativa":"Gôndola com vazio alto/ruptura detectada"}]
    return {"items": itens}

# ==== Finalização de visita e criação de pedido ====
def _finalize_visit_and_create_order() -> bool:
    vid = st.session_state.active_visit_id
    if not vid:
        st.error("Nenhuma visita ativa para finalizar.")
        return False
    sug = _build_purchase_suggestions()
    items = sug.get("items", [])
    order_needed = bool(items)
    df_visit = sql_fetch_df("SELECT * FROM visits WHERE id=?", (vid,))
    if df_visit.empty:
        st.error("Dados da visita não encontrados.")
        return False
    visit_data = df_visit.iloc[0]
    pdf_path = None
    if order_needed:
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        items_json = json.dumps(items, ensure_ascii=False)
        sql_execute(
            """INSERT INTO orders (visit_id, id_cliente, loja, filial, promotor_id, promotor_nome, created_ts, items_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(vid), visit_data["id_cliente"], visit_data["loja"], visit_data["filial"], 
             visit_data["promotor_id"], visit_data["promotor_nome"], now_ts, items_json, "PENDENTE")
        )
        pdf_text = "Sugestão de compra (automática):\n" + "\n".join([f"- {i.get('produto','?')} — {i.get('quantidade','?')} un. ({i.get('justificativa','')})" for i in items])
        pdf_path = _save_purchase_pdf(visit_data["loja"], visit_data["filial"], visit_data["promotor_nome"], pdf_text)
    result_json_data = {
        "final_review": {
            "level": st.session_state.final_review_level, "text": st.session_state.final_review_text,
            "empty_before": st.session_state.empty_before, "empty_after": st.session_state.empty_after, "empty_delta": st.session_state.empty_delta
        },
        "purchase_pdf_path": pdf_path,
    }
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur = minutes_between(visit_data["checkin_ts"], now)
    sql_execute("""UPDATE visits
                   SET result_json=?, checklist_json=?, checkout_ts=?, duracao_min=?, eficiencia=?, purchase_pdf_path=?
                   WHERE id=?""",
                (json.dumps(result_json_data, ensure_ascii=False),
                 json.dumps(st.session_state.guided.get("steps", []), ensure_ascii=False),
                 now, dur, 1.0, pdf_path, vid))
    st.session_state.active_visit_id = None
    return order_needed

# ==== Helpers de dados ====
def _period_bounds(periodo: str) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    today = datetime.date.today()
    if periodo == "Hoje": return today, today
    if periodo == "Semana": return today - datetime.timedelta(days=today.weekday()), today
    if periodo == "Mês": return today.replace(day=1), today
    return None, None

def visits_period_df(periodo: str, filial: str|None=None, promotor_id: str|None=None) -> pd.DataFrame:
    df = sql_fetch_df("SELECT * FROM visits")
    if df.empty: return df
    df["day"] = pd.to_datetime(df["checkin_ts"], errors="coerce").dt.date
    start, end = _period_bounds(periodo)
    if start: df = df[df["day"] >= start]
    if end: df = df[df["day"] <= end]
    if filial: df = df[df["filial"] == filial]
    if promotor_id: df = df[df["promotor_id"] == promotor_id]
    return df.copy()

def promoters_from_users(filial: str | None = None) -> pd.DataFrame:
    q = "SELECT id, username, filial, promotor_id, promotor_nome, is_active FROM users WHERE role='promotor'"
    df = sql_fetch_df(q)
    if filial: df = df[df["filial"] == filial]
    return df

def promoter_status_table(periodo: str, filial: str | None) -> pd.DataFrame:
    users = promoters_from_users(filial)
    vt = visits_period_df(periodo, filial)
    if not vt.empty:
        agg = vt.groupby("promotor_id").agg(
            visitas=("id","count"),
            concluidas=("checkout_ts", lambda s: s.notna().sum())
        ).reset_index()
    else:
        agg = pd.DataFrame(columns=["promotor_id","visitas","concluidas"])
    out = users.merge(agg, how="left", on="promotor_id")
    out["visitas"] = out["visitas"].fillna(0).astype(int)
    out["concluidas"] = out["concluidas"].fillna(0).astype(int)
    out["status_periodo"] = out["visitas"].apply(lambda x: "✅ realizou" if x>0 else "⭕ não realizou")
    out = out.rename(columns={"username":"usuário","promotor_nome":"promotor","promotor_id":"id_promotor"})
    cols = ["usuário","promotor","filial","id_promotor","status_periodo","visitas","concluidas","is_active"]
    return out[cols].sort_values(["filial","promotor"])

# ==== Helpers de Pedidos e Integração ====
STATUS_OPTIONS = ["PENDENTE", "ENVIADO", "ATENDIDO", "CANCELADO"]
def fetch_orders_df(periodo: str, filial: str|None=None, promotor_id: str|None=None) -> pd.DataFrame:
    df = sql_fetch_df("SELECT * FROM orders ORDER BY created_ts DESC")
    if df.empty: return df
    df["day"] = pd.to_datetime(df["created_ts"], errors="coerce").dt.date
    start, end = _period_bounds(periodo)
    if start: df = df[df["day"] >= start]
    if end: df = df[df["day"] <= end]
    if filial: df = df[df["filial"] == filial]
    if promotor_id: df = df[df["promotor_id"] == promotor_id]
    return df.copy()

def update_order_status(order_id: int, new_status: str):
    if new_status in STATUS_OPTIONS:
        sql_execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order_id))

def delete_order(order_id: int):
    sql_execute("DELETE FROM orders WHERE id = ?", (order_id,))

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def _parse_product_id(product_name: str) -> str:
    match = re.search(r'\(id[:\s]*(\d+)\)', product_name, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'(\d+)$', product_name)
    if match:
        return match.group(1)
    return "PRODUTO_NAO_ENCONTRADO"

def fetch_client_cnpj_by_id(client_id: str, filial: str) -> Optional[str]:
    """
    Busca o CPF/CNPJ de um cliente no banco de dados MySQL usando o ID interno.
    """
    if not client_id or not filial:
        return None
    try:
        schema = schema_from_filial(filial)
        engine = get_engine_for_schema(schema)
        with engine.connect() as conn:
            query = text("SELECT cpf_cnpj FROM cliente WHERE id = :id LIMIT 1")
            result = conn.execute(query, {"id": client_id}).scalar_one_or_none()
            return str(result).strip() if result else None
    except Exception as e:
        st.warning(f"Erro ao buscar CNPJ do cliente ID {client_id} na filial {filial}: {e}")
        return None

def send_order_to_clube_venda(order_id: int) -> Tuple[bool, str, Optional[str], Optional[dict]]:
    df_order = sql_fetch_df("SELECT * FROM orders WHERE id = ?", (order_id,))
    if df_order.empty:
        return False, "Pedido não encontrado no banco de dados local.", None, None
    order = df_order.iloc[0]

    filial = order['filial']
    internal_client_id = order['id_cliente']
    token = CLUBE_VENDA_API_TOKENS.get(filial.upper())

    if not CLUBE_VENDA_API_URL or not token:
        return False, f"URL da API ou Token para a filial '{filial}' não configurado.", None, None

    client_cnpj = fetch_client_cnpj_by_id(internal_client_id, filial)
    if not client_cnpj:
        return False, f"Não foi possível encontrar o CNPJ/CPF para o cliente com ID interno '{internal_client_id}'. Verifique o cadastro do cliente.", None, None

    payload_dict = None
    headers = None
    try:
        items_from_db = json.loads(order["items_json"])
        
        itens_pedido_api = []
        total_pedido = 0.0
        for item in items_from_db:
            try:
                preco_unitario = float(item.get("precoUnitario", 0.01))
            except (ValueError, TypeError):
                preco_unitario = 0.01
            
            try:
                quantidade = int(item.get("quantidade", 0))
            except (ValueError, TypeError):
                quantidade = 0
            
            total_item = preco_unitario * quantidade
            total_pedido += total_item
            
            produto_id = item.get("produto_id")
            if not produto_id: 
                produto_id = _parse_product_id(item.get("produto", ""))

            itens_pedido_api.append({
                "percentualDm": 0,
                "precoUnitario": preco_unitario,
                "quantidade": quantidade,
                "totalDesconto": 0,
                "totalItem": round(total_item, 2),
                "valorTrocaNaoAutorizada": 0,
                "produto": { "id": str(produto_id) }
            })

        payload_dict = {
            "dataPedido": datetime.datetime.now().strftime("%Y-%m-%d"),
            "prazoMedio": 0,
            "totalDesconto": 0,
            "custoPedido": 0,
            "saldoDm": 0,
            "margem": 0,
            "totalPedido": round(total_pedido, 2),
            "sincronizado": 0,
            "cliente": { "id": client_cnpj },
            "notificarTroca": 0,
            "trocaEnviada": 0,
            "enviarEmail": 0,
            "emailEnviado": 0,
            "confirmado": 0,
            "ecommerce": 0,
            "origem": "ECOMMERCE",
            "itensPedido": itens_pedido_api
        }
        
        err = validate_payload(payload_dict)
        if err:
            return False, f"Falha na validação do payload: {err}", json.dumps(payload_dict, indent=2, ensure_ascii=False), None
        
        auth_string = f"1:{token}"
        auth_bytes = auth_string.encode('ascii')
        b64_auth = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json;charset=UTF-8",
            "accept": "application/json;charset=UTF-8"
        }
        
        payload_str = json.dumps(payload_dict, ensure_ascii=False)
        response = requests.post(CLUBE_VENDA_API_URL, headers=headers, data=payload_str.encode('utf-8'), timeout=30)
        
        if 200 <= response.status_code < 300:
            return True, f"Pedido {order_id} enviado com sucesso!", None, None
        else:
            return False, f"Falha ao enviar pedido {order_id}. Status: {response.status_code}, Resposta: {response.text}", payload_str, headers

    except Exception as e:
        error_payload_str = json.dumps(payload_dict, indent=2, ensure_ascii=False) if payload_dict else "Falha ao montar o payload."
        return False, f"Erro inesperado ao processar o pedido {order_id}: {e}", error_payload_str, headers


# ==== Gerador de Relatório Detalhado de Visita ====
def _generate_detailed_visit_report(visit_id: int) -> Tuple[Optional[str], str]:
    df = sql_fetch_df("SELECT * FROM visits WHERE id = ?", (visit_id,))
    if df.empty:
        return None, "Visita não encontrada."
    
    visit = df.iloc[0]
    
    content = []
    content.append(f"Relatório Detalhado da Visita #{visit['id']}")
    content.append("="*40)
    content.append(f"Loja: {visit.get('loja', 'N/A')}")
    content.append(f"Cliente ID: {visit.get('id_cliente', 'N/A')}")
    content.append(f"Filial: {visit.get('filial', 'N/A')}")
    content.append(f"Promotor: {visit.get('promotor_nome', 'N/A')} (ID: {visit.get('promotor_id', 'N/A')})")
    content.append(f"Check-in: {visit.get('checkin_ts', 'N/A')}")
    content.append(f"Check-out: {visit.get('checkout_ts', 'N/A')}")
    content.append(f"Duração: {visit.get('duracao_min', 'N/A')} minutos")
    content.append("\n" + "-"*40 + "\n")
    content.append("Análise da Gôndola (IA):")
    content.append(visit.get('analysis_gondola') or "Nenhuma análise de gôndola registrada.")
    content.append("\n" + "-"*40 + "\n")
    content.append("Análise do Estoque (IA):")
    content.append(visit.get('analysis_estoque') or "Nenhuma análise de estoque registrada.")
    content.append("\n" + "-"*40 + "\n")
    content.append("Checklist Executado:")
    try:
        checklist = json.loads(visit.get('checklist_json', '[]'))
        if checklist:
            for i, step in enumerate(checklist): content.append(f"{i+1}. {step}")
        else: content.append("Nenhum checklist gerado.")
    except Exception: content.append("Erro ao carregar o checklist.")
    content.append("\n" + "-"*40 + "\n")
    content.append("Revisão Final (Antes x Depois):")
    try:
        result = json.loads(visit.get('result_json', '{}'))
        final_review = result.get('final_review', {})
        if final_review and final_review.get('text'):
            content.append(f"Nível: {final_review.get('level', 'N/A')}")
            content.append(f"Análise: {final_review.get('text', 'N/A')}")
            content.append(f"Vazio Antes: ~{final_review.get('empty_before', 'N/A')}% | Vazio Depois: ~{final_review.get('empty_after', 'N/A')}%")
        else: content.append("Nenhuma revisão final registrada.")
    except Exception: content.append("Erro ao carregar a revisão final.")

    full_report_text = "\n".join(content)
    
    out_dir = os.path.join(DATA_DIR, "reports_gerados"); os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_loja = re.sub(r"[^\w]+", "_", visit.get('loja', 'loja'), flags=re.UNICODE)[:32]
    fname = f"relatorio_visita_{visit['id']}_{safe_loja}_{ts}.pdf"
    path = os.path.join(out_dir, fname)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4
        text_obj = c.beginText(2*cm, h - 2*cm)
        text_obj.setFont("Helvetica", 9)
        
        for line in full_report_text.splitlines():
            words = line.split()
            current_line = ""
            for word in words:
                if c.stringWidth(current_line + " " + word, "Helvetica", 9) < (w - 4*cm):
                    current_line += " " + word
                else:
                    text_obj.textLine(current_line.strip())
                    current_line = word
            text_obj.textLine(current_line.strip())
            if text_obj.getY() < 3*cm:
                c.drawText(text_obj)
                c.showPage()
                text_obj = c.beginText(2*cm, h - 2*cm)
                text_obj.setFont("Helvetica", 9)

        c.drawText(text_obj)
        c.save()
        return path, full_report_text
    except Exception as e:
        txt_path = path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f: f.write(full_report_text)
        return txt_path, f"Falha ao gerar PDF ({e}), foi gerado um .txt como alternativa."

# ==== Mapa ====
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
    if df_map.empty: return df_map
    df_map["lat"] = pd.to_numeric(df_map.get("latitude"), errors="coerce")
    df_map["lon"] = pd.to_numeric(df_map.get("longitude"), errors="coerce")
    df_map = df_map.dropna(subset=["lat","lon"]).copy()
    if df_map.empty: return df_map
    done_ids = []
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

# ==== Telas ====
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
        st.session_state.auth = {"id":urow["id"],"username":urow["username"],"role":urow["role"],
                                 "filial":urow.get("filial") or "","promotor_id":urow.get("promotor_id") or "",
                                 "promotor_nome":urow.get("promotor_nome") or ""}
        st.session_state.screen = "home"; st.rerun()

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
            st.markdown("### Ações")
            col1, col2, col3 = st.columns(3)
            with col1: uid = st.number_input("ID usuário", min_value=1, step=1)
            with col2:
                if st.button("Ativar"): user_set_active(int(uid), True); st.rerun()
                if st.button("Inativar"): user_set_active(int(uid), False); st.rerun()
            with col3:
                nova = st.text_input("Nova senha")
                if st.button("Resetar senha"):
                    if nova: user_reset_password(int(uid), nova); st.success("Senha atualizada."); st.rerun()
                    else: st.warning("Informe a nova senha.")
    with tabs[1]:
        filial_sel = st.selectbox("Filial (para promotor)", FILIAIS, index=FILIAIS.index("TPH") if "TPH" in FILIAIS else 0, key="admin_filial_sel")
        with st.form("f_new_user"):
            colA, colB = st.columns(2)
            with colA:
                username = st.text_input("Usuário")
                role = st.selectbox("Perfil", ["promotor","admin"], index=0)
                senha = st.text_input("Senha temporária", type="password")
            with colB:
                st.text_input("Filial Selecionada", value=filial_sel, disabled=True)
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

        manual = False
        if df_rot.empty:
            st.warning("Nenhuma loja para HOJE.")
            manual = True
        else:
            opcoes = df_rot.apply(lambda r: f"{r['id_cliente']} — {r.get('nome_cliente','(sem nome)')}", axis=1).tolist()
            i = st.selectbox("Loja do roteiro (HOJE)", list(range(len(opcoes))), format_func=lambda k: opcoes[k])
            row = df_rot.iloc[i]
            st.session_state.loja_id = str(row["id_cliente"])
            st.session_state.loja_nome = str(row.get("nome_cliente") or row.get("razao_social") or st.session_state.loja_id)

        with st.expander("Visita fora da programação (manual)"):
            manual = st.checkbox("Realizar visita fora da programação")
            if manual:
                st.session_state.loja_id = st.text_input("ID do cliente (manual)", value=st.session_state.get("loja_id",""))
                st.session_state.loja_nome = st.text_input("Nome da loja (manual)", value=st.session_state.get("loja_nome",""))

        if st.button("✅ Confirmar check-in", type="primary", use_container_width=True, disabled=not (st.session_state.loja_id)):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql_execute("""INSERT INTO visits (id_cliente, loja, filial, promotor_id, promotor_nome, checkin_ts)
                           VALUES (?,?,?,?,?,?)""",
                        (st.session_state.loja_id, st.session_state.loja_nome or st.session_state.loja_id,
                         filial, promotor_id, promotor_nome, ts))
            vid = sql_fetch_df("SELECT last_insert_rowid() AS id").iloc[0,0]
            st.session_state.active_visit_id = int(vid)
            st.success(f"Check-in: **{st.session_state.loja_nome or st.session_state.loja_id}**.")
            go("gondola")
    else: # ADMIN
        colA, colB, colC = st.columns([1.1, 1.6, 1.6])
        with colA:
            filial = st.selectbox("Filial", FILIAIS, index=FILIAIS.index("TPH") if "TPH" in FILIAIS else 0)
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

        with st.expander("Visita fora da programação (manual)"):
            manual = st.checkbox("Realizar visita fora da programação (admin)")
            if manual:
                st.session_state.loja_id = st.text_input("ID do cliente (manual)", value=st.session_state.get("loja_id",""))
                st.session_state.loja_nome = st.text_input("Nome da loja (manual)", value=st.session_state.get("loja_nome",""))

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
        st.caption("Informe seus produtos para focar a IA. Cole um por linha e/ou envie CSV/Excel (primeira coluna).")
        txt_cat = st.text_area("Lista (um por linha)", value="\n".join(st.session_state.get("catalogo_produtos", [])), height=140)
        up_cat = st.file_uploader("CSV/Excel", type=["csv","xlsx","xls"], key="up_catalogo")
        if st.button("💾 Salvar catálogo p/ esta visita"):
            lista = parse_catalog_input(txt_cat, up_cat)
            st.session_state.catalogo_produtos = lista
            if st.session_state.active_visit_id:
                sql_execute("UPDATE visits SET catalogo_json=? WHERE id=?",
                            (json.dumps(lista, ensure_ascii=False), st.session_state.active_visit_id))
            st.success(f"{len(lista)} itens salvos.")
    cap = st.camera_input("📸 Tirar foto da GÔNDOLA", key=f"cam_g_{st.session_state.cam_gondola_key}")
    if cap: add_captured_photo("imgs_gondola", cap)
    if st.button("➕ Outra foto"): st.session_state.cam_gondola_key += 1; st.rerun()
    if st.session_state.imgs_gondola: st.image([x["bytes"] for x in st.session_state.imgs_gondola], width=220)
    st.button("➡️ Ir para Estoque", type="primary", on_click=lambda: go("estoque"),
              disabled=len(st.session_state.imgs_gondola)==0, use_container_width=True)
    st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)

def screen_estoque():
    require_auth()
    st.header("📦 Estoque — fotos por câmera")
    if len(st.session_state.imgs_gondola)==0:
        st.warning("Antes, capture pelo menos 1 foto da GÔNDOLA."); return

    no_estoque_checkbox = st.checkbox("Loja NÃO possui estoque físico", key="no_estoque")
    cap = st.camera_input("📸 Tirar foto do ESTOQUE", key=f"cam_e_{st.session_state.cam_estoque_key}")
    if cap: add_captured_photo("imgs_estoque", cap)
    cols = st.columns(2)
    with cols[0]:
        if st.button("➕ Outra foto"): st.session_state.cam_estoque_key += 1; st.rerun()
    with cols[1]:
        st.button("⬅️ Voltar", on_click=lambda: go("gondola"), use_container_width=True)

    if st.session_state.imgs_estoque:
        st.image([x["bytes"] for x in st.session_state.imgs_estoque], width=220)

    st.button("✨ Gerar Checklist", type="primary", on_click=lambda: generate_combined_checklist(),
              disabled=(len(st.session_state.imgs_estoque)==0 and not no_estoque_checkbox),
              use_container_width=True)

def generate_combined_checklist():
    require_auth()
    imgs_g = photos_to_pil("imgs_gondola"); imgs_e = photos_to_pil("imgs_estoque")
    no_estoque = bool(st.session_state.get("no_estoque", False))

    with st.spinner("IA analisando gôndola..."): ag = analyze_gondola(imgs_g) if imgs_g else "Sem fotos da gôndola."
    if no_estoque:
        ae = "Loja marcada como **sem estoque físico**. Considerar reposição baseada em giro e ruptura na gôndola."
    else:
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
        st.button("⬅️ Voltar para o Início", on_click=lambda: go("home"), use_container_width=True)
        return

    total = len(steps)
    atual = steps[idx]
    st.markdown("### ✅ Checklist guiado (um passo por vez)")
    st.progress(idx/total if total else 0, text=f"Etapa {idx+1} de {total}")
    st.markdown(f"**Passo atual:** {atual}")

    # --- INÍCIO DA ATUALIZAÇÃO ---
    audio_placeholder = st.empty()

    if st.session_state.audio_to_play:
        audio_placeholder.audio(st.session_state.audio_to_play['bytes'], format=st.session_state.audio_to_play['mime'])
        st.session_state.audio_to_play = None

    if st.button("🔊 Ouvir instrução", use_container_width=True):
        with st.spinner("Gerando áudio..."):
            audio_bytes, mime = tts_bytes(atual)
            if audio_bytes:
                st.session_state.audio_to_play = {"bytes": audio_bytes, "mime": mime}
                st.rerun()
    # --- FIM DA ATUALIZAÇÃO ---

    st.markdown("Confirme (ex.: 'ok', 'feito') ou registre observação:")
    saida_txt = st.text_input("", key=f"txt_{idx}")
    ok_text = st.button("✅ Confirmar por texto", use_container_width=True)

    saida_voice = ""
    ok_voice_clicked = False
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

    colX, colY = st.columns(2)
    with colX:
        if st.button("⏭️ Pular passo", use_container_width=True): avancar = True
    with colY:
        st.button("⬅️ Voltar para o Início", on_click=lambda: go("home"), use_container_width=True)

    if avancar:
        novo_idx = idx + 1
        if novo_idx >= total:
            st.success("Checklist concluído!")
            st.info("Indo para a revisão final para concluir a visita.")
            go("audit_final")
            st.rerun()
        else:
            st.session_state.guided["idx"] = novo_idx
            st.rerun()


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

    if st.session_state.get("visit_just_finalized"):
        st.success("Visita finalizada com sucesso! Cliente sem recomendação de compras no momento.")
        st.info("Você pode voltar ao início para uma nova visita ou ver o dashboard.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🏠 Voltar para o Início", use_container_width=True):
                st.session_state.visit_just_finalized = False
                go("home")
                st.rerun()
        with c2:
            if st.button("📊 Ir para o Dashboard", use_container_width=True, type="primary"):
                st.session_state.visit_just_finalized = False
                go("dashboard")
                st.rerun()
        return

    st.header("🏁 Revisão final e Geração de Pedido")
    st.info("Tire uma foto final da gôndola para registrar o resultado do seu trabalho e clique em 'Gerar Pedido' para finalizar.")
    if not st.session_state.imgs_gondola:
        st.info("Faça pelo menos 1 foto da gôndola antes."); return
    
    cap = st.camera_input("📸 Foto FINAL da gôndola", key=f"cam_f_{st.session_state.cam_final_key}")
    if cap: st.session_state.final_gondola = cap.getvalue()
    if st.button("🔁 Nova final"): st.session_state.cam_final_key += 1; st.rerun()
    
    can = bool(st.session_state.final_gondola)
    if st.button("🔎 Analisar Antes x Depois", type="secondary", disabled=not can, use_container_width=True):
        before = photos_to_pil("imgs_gondola"); after = Image.open(io.BytesIO(st.session_state.final_gondola)).convert("RGB")
        with st.spinner("IA analisando o antes e depois..."):
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

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧾 Gerar Pedido e Finalizar Visita", type="primary", use_container_width=True, disabled=st.session_state.active_visit_id is None):
            with st.spinner("Finalizando visita e gerando pedido..."):
                order_created = _finalize_visit_and_create_order()
            
            if order_created:
                st.success("Pedido gerado com sucesso!")
                go("pedidos")
            else:
                st.session_state.visit_just_finalized = True
            st.rerun()
    with c2:
        st.button("⬅️ Voltar para o Início", on_click=lambda: go("home"), use_container_width=True)

def screen_pedidos():
    require_auth()
    st.header("📋 Painel de Pedidos")

    auth = st.session_state.auth
    
    with st.expander("➕ Criar Pedido Manual"):
        if auth['role'] == 'admin':
            filial_manual = st.selectbox("Filial do Pedido", FILIAIS, key="manual_filial_select")
        else:
            filial_manual = auth['filial']

        if filial_manual:
            engine_schema = get_engine_for_schema(schema_from_filial(filial_manual))
            products_df = fetch_products(engine_schema)
        else:
            products_df = pd.DataFrame(columns=['codigo', 'descricao'])

        if auth['role'] == 'admin':
            promotores_df = promoters_from_users(filial_manual)
            if not promotores_df.empty:
                promotor_opts = {f"{r['promotor_id']}|{r['promotor_nome']}": f"{r['promotor_nome']}" for _, r in promotores_df.iterrows()}
                promotor_sel = st.selectbox("Promotor Responsável", promotor_opts.keys(), format_func=lambda x: promotor_opts[x], key="manual_promotor")
                promotor_id, promotor_nome = promotor_sel.split('|')
            else:
                st.warning("Nenhum promotor cadastrado para esta filial.")
                promotor_id, promotor_nome = None, None
        else:
            promotor_id = auth['promotor_id']
            promotor_nome = auth['promotor_nome']
            st.info(f"Pedido a ser criado por: **{promotor_nome}** (Filial: **{filial_manual}**)")

        id_cliente_manual = None
        loja_manual = None
        clients_df = pd.DataFrame()
        if promotor_id:
            clients_df = fetch_all_clients_for_promoter(engine_schema, promotor_id)
        
        if not clients_df.empty:
            client_options = {f"{row['id']}|{row['nome_cliente']}": f"{row['nome_cliente']} (ID: {row['id']})" for _, row in clients_df.iterrows()}
            client_list = ["Selecione um cliente"] + list(client_options.keys())
            
            selected_client_key = st.selectbox("Selecione o Cliente", client_list, format_func=lambda x: client_options.get(x, "Selecione um cliente"))

            if selected_client_key != "Selecione um cliente":
                id_cliente_manual, loja_manual = selected_client_key.split('|', 1)
        else:
            st.warning("Nenhum cliente associado a este promotor. Insira manualmente.")
            id_cliente_manual = st.text_input("ID do Cliente (CNPJ/CPF)")
            loja_manual = st.text_input("Nome da Loja")


        st.markdown("---")
        st.subheader("Itens do Pedido")

        if not products_df.empty:
            product_options = {f"{row['codigo']}": f"{row['descricao']}" for _, row in products_df.iterrows()}
            product_list = ["Selecione um produto"] + list(product_options.keys())
            
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1.5, 2, 1])
            with col1:
                selected_product_id = st.selectbox("Pesquisar Produto", product_list, format_func=lambda x: product_options.get(x, "Selecione um produto"), key="manual_product")
            with col2:
                qty = st.number_input("Qtd.", min_value=1, step=1, key="manual_qty")
            with col3:
                price = st.number_input("Preço Unit.", min_value=0.0, step=0.01, format="%.2f", key="manual_price")
            with col4:
                just = st.text_input("Justificativa", key="manual_just")
            with col5:
                st.write("") 
                if st.button("Adicionar", use_container_width=True):
                    if selected_product_id != "Selecione um produto":
                        st.session_state.manual_order_items.append({
                            "produto_id": selected_product_id,
                            "produto": product_options[selected_product_id],
                            "quantidade": qty,
                            "precoUnitario": price,
                            "justificativa": just or "Pedido Manual"
                        })
                    else:
                        st.warning("Por favor, selecione um produto.")
        else:
            st.warning("Não foi possível carregar os produtos. Verifique a conexão com o banco ou a seleção da filial.")

        if st.session_state.manual_order_items:
            st.dataframe(st.session_state.manual_order_items, use_container_width=True)
            if st.button("Remover último item", type="secondary"):
                st.session_state.manual_order_items.pop()
                st.rerun()

        if st.button("Criar Pedido Manual", type="primary", use_container_width=True):
            if not all([id_cliente_manual, loja_manual, promotor_id]) or not st.session_state.manual_order_items:
                st.error("Preencha os dados do cliente e adicione pelo menos um item ao pedido.")
            else:
                sql_execute(
                    """INSERT INTO orders (visit_id, id_cliente, loja, filial, promotor_id, promotor_nome, created_ts, items_json, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (None, id_cliente_manual, loja_manual, filial_manual, promotor_id, promotor_nome, 
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(st.session_state.manual_order_items, ensure_ascii=False), "PENDENTE")
                )
                st.success(f"Pedido manual para a loja '{loja_manual}' criado com sucesso!")
                st.session_state.manual_order_items = []
                st.rerun()
    
    promotor_id_filtro = None
    filial_filtro = None

    if auth["role"] == "promotor":
        promotor_id_filtro = auth["promotor_id"]
        filial_filtro = auth["filial"]
        periodo = seg_ctrl("Período", ["Hoje", "Semana", "Mês", "Tudo"], "Hoje")
    else: # Admin
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            periodo = st.selectbox("Período", ["Hoje", "Semana", "Mês", "Tudo"])
        with c2:
            filiais_opts = ["Todas"] + FILIAIS
            filial_sel = st.selectbox("Filial", filiais_opts)
            filial_filtro = None if filial_sel == "Todas" else filial_sel
        with c3:
            promotores = promoters_from_users(filial_filtro)
            if not promotores.empty:
                promotor_opts = {r['promotor_id']: f"{r['promotor_nome']} ({r['filial']})" for _, r in promotores.iterrows()}
                promotor_opts_list = ["Todos"] + list(promotor_opts.keys())
                promotor_sel = st.selectbox("Promotor", promotor_opts_list, format_func=lambda x: "Todos" if x == "Todos" else promotor_opts[x])
                promotor_id_filtro = None if promotor_sel == "Todos" else promotor_sel
            else:
                st.info("Nenhum promotor para a filial selecionada.")

    df_orders = fetch_orders_df(periodo, filial=filial_filtro, promotor_id=promotor_id_filtro)

    if df_orders.empty:
        st.info("Nenhum pedido encontrado para os filtros selecionados.")
        return

    st.markdown("---")
    cols = st.columns(len(STATUS_OPTIONS))
    
    for i, status in enumerate(STATUS_OPTIONS):
        with cols[i]:
            st.markdown(f"<h5 style='text-align: center;'>{status}</h5>", unsafe_allow_html=True)
            df_status = df_orders[df_orders['status'] == status]
            for _, order in df_status.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius: 7px; padding: 10px; margin-bottom: 10px; background-color: #f9f9f9;">
                        <small>Pedido #{order['id']}</small><br>
                        <strong>{order['loja']}</strong><br>
                        <small>{order['promotor_nome']} em {pd.to_datetime(order['created_ts']).strftime('%d/%m/%Y')}</small>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Ver detalhes"):
                        try:
                            items = json.loads(order['items_json'])
                            df_items = pd.DataFrame(items)
                            st.dataframe(df_items, use_container_width=True)
                            
                            csv = convert_df_to_csv(df_items)
                            st.download_button(
                                label="📄 Baixar CSV", data=csv,
                                file_name=f"pedido_{order['id']}_{order['loja']}.csv", mime="text/csv",
                                key=f"csv_{order['id']}"
                            )
                        except Exception:
                            st.error("Não foi possível carregar os itens do pedido.")

                        new_status = st.selectbox("Mudar Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(order['status']), key=f"status_{order['id']}")
                        if st.button("Atualizar Status", key=f"update_{order['id']}"):
                            update_order_status(order['id'], new_status)
                            st.success("Status atualizado!")
                            st.rerun()
                        
                        if order['status'] == 'PENDENTE':
                            if st.button("📦 Enviar ao ERP", key=f"erp_{order['id']}", type="primary"):
                                with st.spinner("Enviando pedido para a API..."):
                                    success, message, payload_str, headers_sent = send_order_to_clube_venda(order['id'])
                                if success:
                                    st.success(message)
                                    update_order_status(order['id'], "ENVIADO")
                                    st.rerun()
                                else:
                                    st.error(message)
                                    if payload_str:
                                        st.subheader("Dados Enviados (para depuração):")
                                        if headers_sent:
                                            st.text("Cabeçalhos (Headers):")
                                            st.json(headers_sent)
                                        st.text("Corpo (Payload):")
                                        st.code(payload_str, language='json')
                        
                        if auth['role'] == 'admin':
                            if st.button("🗑️ Excluir Pedido", key=f"delete_{order['id']}", type="secondary", use_container_width=True, help="Esta ação é irreversível."):
                                delete_order(order['id'])
                                st.success(f"Pedido #{order['id']} excluído com sucesso.")
                                st.rerun()


# ==== Consulta e Geração de Relatórios de Visitas ====
def screen_consulta_visitas():
    require_auth()
    st.header("🔍 Consulta e Geração de Relatórios de Visitas")
    st.info("Use os filtros para encontrar uma visita específica e gerar um relatório detalhado a qualquer momento.")

    auth = st.session_state.auth
    promotor_id_filtro = None
    filial_filtro = None

    if auth["role"] == "promotor":
        promotor_id_filtro = auth["promotor_id"]
        filial_filtro = auth["filial"]
        periodo = seg_ctrl("Período", ["Hoje", "Semana", "Mês", "Tudo"], "Hoje")
    else: # Admin
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            periodo = st.selectbox("Período", ["Hoje", "Semana", "Mês", "Tudo"])
        with c2:
            filiais_opts = ["Todas"] + FILIAIS
            filial_sel = st.selectbox("Filial", filiais_opts)
            filial_filtro = None if filial_sel == "Todas" else filial_sel
        with c3:
            promotores = promoters_from_users(filial_filtro)
            if not promotores.empty:
                promotor_opts = {r['promotor_id']: f"{r['promotor_nome']} ({r['filial']})" for _, r in promotores.iterrows()}
                promotor_opts_list = ["Todos"] + list(promotor_opts.keys())
                promotor_sel = st.selectbox("Promotor", promotor_opts_list, format_func=lambda x: "Todos" if x == "Todos" else promotor_opts[x])
                promotor_id_filtro = None if promotor_sel == "Todos" else promotor_sel

    df_visits = visits_period_df(periodo, filial_filtro, promotor_id_filtro)
    
    if df_visits.empty:
        st.info("Nenhuma visita encontrada para os filtros selecionados.")
        return

    st.markdown("---")
    st.dataframe(df_visits[['id', 'checkin_ts', 'loja', 'promotor_nome', 'filial', 'duracao_min']].rename(
        columns={'id':'ID', 'checkin_ts':'Data', 'loja':'Loja', 'promotor_nome':'Promotor', 'filial':'Filial', 'duracao_min':'Duração (min)'}
    ), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Gerar Relatório Detalhado de Visita")
    
    visit_ids = df_visits['id'].tolist()
    if not visit_ids:
        return
        
    selected_id = st.selectbox("Selecione o ID da visita para gerar o relatório", options=visit_ids)

    if st.button("Gerar Relatório", type="primary", use_container_width=True):
        with st.spinner(f"Gerando relatório para a visita #{selected_id}..."):
            pdf_path, report_content = _generate_detailed_visit_report(selected_id)
        
        if pdf_path and os.path.exists(pdf_path):
            st.success("Relatório gerado com sucesso!")
            st.text_area("Prévia do Relatório", report_content, height=300)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Baixar Relatório Completo em PDF",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.error(f"Falha ao gerar o relatório: {report_content}")


def screen_reports():
    require_auth()
    st.header("📤 Arquivo de Relatórios (PDFs)")
    st.info("Esta tela funciona como um arquivo para baixar os PDFs de sugestão de compra que foram gerados no passado.")
    
    periodo = seg_ctrl("Período", ["Hoje","Semana","Mês","Tudo"], "Hoje")
    auth = st.session_state.auth
    if auth["role"] == "promotor":
        df = visits_period_df(periodo, filial=auth["filial"], promotor_id=auth["promotor_id"])
    else:
        colf, colp = st.columns([1,2])
        with colf:
            filial_opt = st.selectbox("Filtrar por filial", ["Todas"] + FILIAIS, index=0, key="rep_filial")
            filial_sel = None if filial_opt == "Todas" else filial_opt
        with colp:
            df_proms = promoters_from_users(filial_sel)
            pro_labels = ["Todos"] + [f"{r['promotor_nome']} (id {r['promotor_id']}) — {r['filial']}" for _, r in df_proms.iterrows()]
            idx = st.selectbox("Filtrar por promotor", list(range(len(pro_labels))), format_func=lambda i: pro_labels[i], key="rep_prom")
            prom_sel_id = None if idx == 0 else str(df_proms.iloc[idx-1]["promotor_id"])
        df = visits_period_df(periodo, filial=filial_sel, promotor_id=prom_sel_id)

    if df.empty or "purchase_pdf_path" not in df.columns:
        st.info("Nenhum relatório PDF encontrado para o período/critério selecionado."); return
    
    df = df[df["purchase_pdf_path"].notna() & (df["purchase_pdf_path"] != "")]
    if df.empty:
        st.info("Nenhum relatório PDF encontrado para o período/critério selecionado."); return

    df = df.sort_values("checkin_ts", ascending=False)
    for _, r in df.iterrows():
        titulo = f"{r.get('checkin_ts','')} — {r.get('loja','(sem loja)')} • Filial {r.get('filial','')}"
        with st.expander(titulo):
            pdf_path = str(r.get("purchase_pdf_path") or "")
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Baixar PDF", f.read(), file_name=os.path.basename(pdf_path), mime="application/pdf", key=f"pdf_{r['id']}")
            else:
                st.warning("Arquivo PDF não encontrado no disco.")


def screen_dashboard():
    require_auth()
    auth = st.session_state.auth
    st.header("📊 Dashboard")
    periodo = seg_ctrl("Período", ["Hoje","Semana","Mês","Tudo"], "Hoje")

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
            try:
                import pydeck as pdk
                layer = pdk.Layer("ScatterplotLayer", data=dfm, get_position='[lon, lat]', get_fill_color='color', get_radius=70, pickable=True)
                view = pdk.ViewState(latitude=float(dfm["lat"].mean()), longitude=float(dfm["lon"].mean()), zoom=10)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html": "<b>{tooltip}</b>", "style": {"color": "white"}}))
            except Exception as e:
                st.warning(f"Mapa indisponível: {e}")
            cols = ["id_cliente","nome_cliente","data_visita","status","visitado_map","latitude","longitude"]
            cols = [c for c in cols if c in dfm.columns]
            st.dataframe(dfm[cols].rename(columns={"visitado_map":"visitado"}), use_container_width=True, hide_index=True)
        st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)
        return

    # ADMIN
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
    if prom_sel_id: df_status = df_status[df_status["id_promotor"].astype(str) == prom_sel_id]
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
        try:
            import pydeck as pdk
            layer = pdk.Layer("ScatterplotLayer", data=dfm, get_position='[lon, lat]', get_fill_color='color', get_radius=70, pickable=True)
            view = pdk.ViewState(latitude=float(dfm["lat"].mean()), longitude=float(dfm["lon"].mean()), zoom=10)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html": "<b>{tooltip}</b>", "style": {"color": "white"}}))
        except Exception as e:
            st.warning(f"Mapa indisponível: {e}")
        st.markdown("#### 📄 Clientes do roteiro (HOJE)")
        cols = ["id_cliente","nome_cliente","data_visita","status","visitado_map","latitude","longitude"]
        cols = [c for c in cols if c in dfm.columns]
        st.dataframe(dfm[cols].rename(columns={"visitado_map":"visitado"}), use_container_width=True, hide_index=True)

    st.button("⬅️ Voltar", on_click=lambda: go("home"), use_container_width=True)

# ==== Sidebar ====
with st.sidebar:
    if st.session_state.auth:
        st.markdown(f"**Usuário:** {st.session_state.auth['username']}  \n**Perfil:** {st.session_state.auth['role']}")
        st.button("🏠 Início", on_click=lambda: go("home"), use_container_width=True)
        st.button("🆕 Nova visita", on_click=lambda: st.session_state.update({
            "loja_id":"", "loja_nome":"", "active_visit_id":None, "imgs_gondola":[], "imgs_estoque":[], 
            "guided":{"steps":[], "idx":0}, "final_gondola":None, "final_review_text":"", "final_review_level":"",
            "empty_before":None, "empty_after":None, "empty_delta":None, "cam_gondola_key":0, "cam_estoque_key":0, 
            "cam_final_key":0, "catalogo_produtos":[], "no_estoque": False, "visit_just_finalized": False
        }) or go("home"), use_container_width=True)
        st.button("📋 Pedidos", on_click=lambda: go("pedidos"), use_container_width=True)
        st.button("🔍 Consultar Visitas", on_click=lambda: go("consulta"), use_container_width=True)
        st.button("📊 Dashboard", on_click=lambda: go("dashboard"), use_container_width=True)
        st.button("📤 Arquivo de Relatórios", on_click=lambda: go("reports"), use_container_width=True, help="Arquivo de PDFs de compra gerados.")
        if st.session_state.auth["role"]=="admin":
            st.button("🛠️ Admin", on_click=lambda: go("admin"), use_container_width=True)
        if st.session_state.active_visit_id:
            if st.button("🔚 Forçar check-out", use_container_width=True, type="secondary"):
                now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dfc=sql_fetch_df("SELECT checkin_ts FROM visits WHERE id=?", (st.session_state.active_visit_id,))
                dur=minutes_between(dfc.loc[0,"checkin_ts"], now) if not dfc.empty else None
                sql_execute("UPDATE visits SET checkout_ts=?, duracao_min=? WHERE id=?",
                            (now, dur, st.session_state.active_visit_id))
                st.success("Check-out concluído."); st.session_state.active_visit_id=None
                st.rerun()
        if st.button("🚪 Sair", on_click=lambda: go("login"), use_container_width=True):
            st.session_state.auth=None; go("login"); st.rerun()
    else:
        st.markdown("### Bem-vindo")
        st.caption("Faça login para continuar.")

# ==== Roteador ====
scr = st.session_state.screen
if scr == "login" and st.session_state.get("auth"):
    go("home"); st.rerun()

if scr=="login":        screen_login()
elif scr=="home":       screen_home()
elif scr=="gondola":     screen_gondola()
elif scr=="estoque":     screen_estoque()
elif scr=="guided":      screen_guided()
elif scr=="audit_final": screen_audit_final()
elif scr=="pedidos":     screen_pedidos()
elif scr=="consulta":    screen_consulta_visitas()
elif scr=="reports":     screen_reports()
elif scr=="dashboard":   screen_dashboard()
elif scr=="admin":       screen_admin()
else:                   screen_login()