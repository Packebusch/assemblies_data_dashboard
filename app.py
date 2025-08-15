import pandas as pd
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import hashlib
import numpy as np
import openai
import time
from datetime import datetime, timedelta

import os, io, requests
DATA = os.getenv("DATA_PATH", "data/recommendations.ndjson")

st.set_page_config(page_title="Bürgerräte – Empfehlungen", layout="wide")
st.title("Bürgerräte – Empfehlungen (Ergebnisse)")

# Rate limiting configuration
MAX_REQUESTS_PER_HOUR = 100  # Maximum API calls per hour per user
MAX_REQUESTS_PER_MINUTE = 20  # Maximum API calls per minute per user

def check_rate_limit():
    """Check if user has exceeded rate limits. Returns (allowed, message)"""
    now = datetime.now()
    
    # Initialize rate limiting in session state
    if "api_calls" not in st.session_state:
        st.session_state.api_calls = []
    
    # Clean old entries (older than 1 hour)
    st.session_state.api_calls = [
        call_time for call_time in st.session_state.api_calls 
        if now - call_time < timedelta(hours=1)
    ]
    
    # Check hourly limit
    if len(st.session_state.api_calls) >= MAX_REQUESTS_PER_HOUR:
        return False, f"Stündliches Limit erreicht ({MAX_REQUESTS_PER_HOUR} Anfragen/Stunde). Versuche es später erneut."
    
    # Check minute limit
    recent_calls = [
        call_time for call_time in st.session_state.api_calls 
        if now - call_time < timedelta(minutes=1)
    ]
    if len(recent_calls) >= MAX_REQUESTS_PER_MINUTE:
        return False, f"Zu viele Anfragen. Warte bitte eine Minute ({MAX_REQUESTS_PER_MINUTE} Anfragen/Minute)."
    
    return True, ""

def record_api_call():
    """Record an API call timestamp"""
    if "api_calls" not in st.session_state:
        st.session_state.api_calls = []
    st.session_state.api_calls.append(datetime.now())

def load_df(path: str):
    # Don't cache this function to avoid DataFrame hashing issues
    df = pd.read_json(path, lines=True)
    for c in ["state_name", "political_level", "policy_area"]:
        if c in df:
            df[c] = df[c].fillna("(unbekannt)")
    # Ensure topic columns exist and are well-typed
    if "primary_topic" not in df:
        df["primary_topic"] = None
    if "secondary_topics" in df:
        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                return [x]
            return []
        df["secondary_topics"] = df["secondary_topics"].apply(_to_list)
    else:
        df["secondary_topics"] = [[] for _ in range(len(df))]
    # Stable id per recommendation (hash of assembly_id + text)
    def _make_id(row):
        base = f"{row.get('assembly_id','')}\n{row.get('recommendation_text','')}"
        return hashlib.sha1(base.encode('utf-8', 'ignore')).hexdigest()
    if "rec_id" not in df:
        try:
            df["rec_id"] = df.apply(_make_id, axis=1)
        except Exception:
            df["rec_id"] = [str(i) for i in range(len(df))]
    return df

df = load_df(DATA)

# Secrets-aware OpenAI config (for Streamlit Cloud)
try:
    OPENAI_API_KEY = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') and hasattr(st.secrets, 'get') else None) or os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = (st.secrets.get("OPENAI_MODEL") if hasattr(st, 'secrets') and hasattr(st.secrets, 'get') else None) or os.getenv("OPENAI_MODEL", "gpt-4o")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

@st.cache_data(show_spinner=False)
def build_index_from_texts(texts: list, rec_ids: list):
    """Build embeddings index from text list, using rec_ids as cache key"""
    if OPENAI_API_KEY is None or not texts:
        return None, None
    
    # Note: This function is cached, so embeddings are only built once per dataset
    # Rate limiting is applied to the chat function instead
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    vecs = []
    for i in range(0, len(texts), 200):
        batch = texts[i:i+200]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        vecs.extend([d.embedding for d in resp.data])
    if not vecs:
        return None, None
    M = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    M = M / norms
    return M

def build_index(df_in: pd.DataFrame):
    """Wrapper function to handle DataFrame caching issues"""
    if OPENAI_API_KEY is None or df_in.empty or "recommendation_text" not in df_in:
        return None, None
    
    # Extract texts and rec_ids for caching
    texts = (df_in["recommendation_text"].fillna("").astype(str)).tolist()
    rec_ids = df_in["rec_id"].tolist() if "rec_id" in df_in.columns else list(range(len(texts)))
    
    # Get cached embeddings
    M = build_index_from_texts(texts, rec_ids)
    if M is None:
        return None, None
    
    # Return embeddings with original DataFrame (no caching issues)
    return M, df_in.reset_index(drop=True)

emb_matrix, df_ix = build_index(df)

with st.sidebar:
    st.header("Filter")
    states = st.multiselect("Bundesland", sorted(df.get("state_name", pd.Series()).dropna().unique()))
    levels = st.multiselect("Ebene", sorted(df.get("political_level", pd.Series()).dropna().unique()))
    # Assembly filter (by title)
    assemblies = st.multiselect(
        "Assembly (Titel)",
        sorted(df.get("assembly_title", pd.Series()).dropna().unique())
    )
    # Year range filter with guard if only a single year exists
    if "start_year" in df and df["start_year"].notna().any():
        y_min, y_max = int(df["start_year"].min()), int(df["start_year"].max())
        if y_min < y_max:
            years = st.slider("Startjahr", y_min, y_max, (y_min, y_max))
        else:
            st.caption(f"Startjahr: {y_min}")
            years = (y_min, y_max)
    else:
        years = (1990, 2030)
    # Topics
    primary_topics = st.multiselect(
        "Thema (Primary)",
        sorted([t for t in df.get("primary_topic", pd.Series()).dropna().unique().tolist() if t])
    )
    # Flatten and collect secondary topic options
    sec_opts = sorted({t for xs in df.get("secondary_topics", pd.Series([[]]*len(df))) for t in (xs or []) if t})
    secondary_topics = st.multiselect("Thema (Secondary)", sec_opts)
    query = st.text_input("Suche in Empfehlung (Volltext)")

f = df.copy()
if states:
    f = f[f["state_name"].isin(states)]
if levels:
    f = f[f["political_level"].isin(levels)]
if assemblies:
    f = f[f["assembly_title"].isin(assemblies)]
if "start_year" in f:
    f = f[(f["start_year"] >= years[0]) & (f["start_year"] <= years[1])]
if query:
    f = f[f["recommendation_text"].str.contains(query, case=False, na=False)]
if primary_topics:
    f = f[f["primary_topic"].isin(primary_topics)]
if secondary_topics and "secondary_topics" in f:
    f = f[f["secondary_topics"].apply(lambda xs: any(t in (xs or []) for t in secondary_topics))]

col1, col2, col3 = st.columns(3)
col1.metric("Empfehlungen", len(f))
col2.metric("Assemblies", f.get("assembly_id", pd.Series()).nunique())
col3.metric("Länder", f.get("state_name", pd.Series()).nunique())

st.subheader("Empfehlungen nach Bundesland / Thema")
if not f.empty and "state_name" in f:
    by_state = f.groupby("state_name").size().reset_index(name="n")
    st.plotly_chart(px.bar(by_state, x="state_name", y="n"), use_container_width=True)
if not f.empty and "primary_topic" in f:
    by_topic = f.groupby("primary_topic").size().reset_index(name="n").sort_values("n", ascending=False)
    st.plotly_chart(px.bar(by_topic, x="primary_topic", y="n"), use_container_width=True)

st.subheader("Empfehlungen")
cols = ["rec_id","assembly_title","state_name","political_level","start_year","primary_topic","secondary_topics","recommendation_title","recommendation_text","file_url","source_page_url"]
show = [c for c in cols if c in f.columns]
table_df = f[show].sort_values([c for c in ["state_name","assembly_title"] if c in show]).reset_index(drop=True)
# AgGrid with single-row selection
grid_df = table_df.copy()
gb = GridOptionsBuilder.from_dataframe(grid_df)
# Hide internal id
if "rec_id" in grid_df.columns:
    gb.configure_columns(["rec_id"], hide=True)
# Ensure a visible column gets the checkbox (if the first is hidden, checkbox would disappear)
checkbox_col = None
for c in ["assembly_title", "state_name", "recommendation_title", "recommendation_text"]:
    if c in grid_df.columns:
        checkbox_col = c
        break
if checkbox_col is None:
    # fallback to the first visible column
    checkbox_col = next((c for c in grid_df.columns if c != "rec_id"), grid_df.columns[0])
gb.configure_selection(selection_mode='single', use_checkbox=True, suppressRowClickSelection=False)
gb.configure_column(checkbox_col, headerCheckboxSelection=False, checkboxSelection=True)
gb.configure_grid_options(domLayout='normal')
grid_options = gb.build()
grid = AgGrid(
    grid_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=520,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True
)
sel = grid.get('selected_rows')
if sel is not None and len(sel) > 0:
    sel_row = sel[0]
    # Try to resolve to full record (prefer rec_id, else match on key fields)
    rid = sel_row.get('rec_id')
    detail_dict = None
    if rid and 'rec_id' in table_df.columns and rid in set(table_df['rec_id']):
        detail_dict = table_df.set_index('rec_id').loc[rid].to_dict()
    else:
        try:
            mask = (table_df.get('assembly_title') == sel_row.get('assembly_title')) & \
                   (table_df.get('recommendation_text') == sel_row.get('recommendation_text'))
            match = table_df[mask]
            if not match.empty:
                detail_dict = match.iloc[0].to_dict()
        except Exception:
            pass
    if detail_dict is None:
        # Fallback: use selected row as-is
        detail_dict = sel_row
    st.session_state["detail_row"] = detail_dict
    def _open_dialog():
        show_detail_dialog()
    st.button("Details anzeigen", on_click=_open_dialog)

st.download_button("Export CSV (gefiltert)", table_df.drop(columns=[c for c in ["rec_id"] if c in table_df.columns]).to_csv(index=False).encode("utf-8"), file_name="empfehlungen_export.csv", mime="text/csv")

# Modal detail
@st.dialog("Empfehlung – Details")
def show_detail_dialog():
    row = st.session_state.get("detail_row", {})
    if not row:
        st.write("Keine Daten")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Assembly**: {row.get('assembly_title','')} ({row.get('state_name','')}, {row.get('political_level','')}, {row.get('start_year','')})")
        st.markdown(f"**Thema**: {row.get('primary_topic','')}")
        st.markdown(f"**Weitere Themen**: {', '.join(row.get('secondary_topics') or [])}")
        if row.get('recommendation_title'):
            st.markdown(f"**Titel**: {row.get('recommendation_title')}")
        st.markdown("**Empfehlungstext**:")
        st.write(row.get('recommendation_text',''))
    with c2:
        if row.get('file_url'):
            st.markdown(f"[Quelldokument]({row.get('file_url')})")
        if row.get('source_page_url'):
            st.markdown(f"[Assembly-Seite]({row.get('source_page_url')})")

# (Removed alternate detail sections to rely on modal from main table)

# --- Floating chat overlay ---
@st.dialog("Chat mit den Empfehlungen")
def show_chat_dialog():
    if OPENAI_API_KEY is None:
        st.info("Setze OPENAI_API_KEY in Secrets, um die Konversation zu aktivieren.")
        return
    
    # Display rate limit status
    if "api_calls" in st.session_state:
        now = datetime.now()
        recent_calls = [
            call_time for call_time in st.session_state.api_calls 
            if now - call_time < timedelta(hours=1)
        ]
        remaining = MAX_REQUESTS_PER_HOUR - len(recent_calls)
        st.caption(f"Verbleibende API-Anfragen diese Stunde: {remaining}/{MAX_REQUESTS_PER_HOUR}")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)
    user_q = st.chat_input("Frage zu den Empfehlungen stellen…", key="chat_input_dialog")
    if not user_q:
        return
    
    # Check rate limit before processing
    allowed, rate_limit_msg = check_rate_limit()
    if not allowed:
        st.error(rate_limit_msg)
        return
    
    with st.chat_message("user"):
        st.write(user_q)
    
    # Record API call for rate limiting
    record_api_call()
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    contexts = []
    if emb_matrix is not None and df_ix is not None and len(emb_matrix) == len(df_ix):
        try:
            q_emb = client.embeddings.create(model="text-embedding-3-small", input=[user_q]).data[0].embedding
            q = np.array(q_emb, dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-9)
            sims = (emb_matrix @ q).astype(np.float32)
            
            # Get all results sorted by similarity (highest first)
            # Filter for relevant results (similarity > threshold)
            similarity_threshold = 0.1  # Adjust this threshold as needed
            relevant_indices = np.where(sims > similarity_threshold)[0]
            
            if len(relevant_indices) > 0:
                # Sort by similarity (highest first)
                sorted_indices = relevant_indices[sims[relevant_indices].argsort()[::-1]]
                
                for i in sorted_indices:
                    row = df_ix.iloc[int(i)]
                    similarity_score = sims[i]
                    ctx = f"- Assembly: {row.get('assembly_title','')} ({row.get('state_name','')}, {row.get('start_year','')})\n  Empfehlung: {row.get('recommendation_text','')}\n  Ähnlichkeit: {similarity_score:.3f}\n  Quelle: {row.get('file_url','')}\n"
                    contexts.append(ctx)
            else:
                # If no results above threshold, take top 10 results anyway
                idxs = sims.argsort()[-10:][::-1]
                for i in idxs:
                    row = df_ix.iloc[int(i)]
                    similarity_score = sims[i]
                    ctx = f"- Assembly: {row.get('assembly_title','')} ({row.get('state_name','')}, {row.get('start_year','')})\n  Empfehlung: {row.get('recommendation_text','')}\n  Ähnlichkeit: {similarity_score:.3f}\n  Quelle: {row.get('file_url','')}\n"
                    contexts.append(ctx)
        except Exception as e:
            st.error(f"Fehler bei der Suche: {e}")
            return
    
    # Intelligent context truncation to stay within token limits
    # Adjust based on model capabilities
    model_context_limits = {
        "gpt-4o-mini": 100000,      # 128k tokens - conservative
        "gpt-4o": 500000,           # 512k tokens - much larger
        "gpt-4-turbo": 500000,      # 512k tokens  
        "gpt-4": 100000,            # 128k tokens
        "gpt-4.1": 800000,          # 1M tokens - huge context
        "gpt-5": 200000             # 256k tokens
    }
    
    max_context_length = model_context_limits.get(OPENAI_MODEL, 100000)
    context_text = chr(10).join(contexts)
    
    # If contexts are too long, truncate intelligently
    if len(context_text) > max_context_length:
        # Take first N contexts that fit within limit
        truncated_contexts = []
        current_length = 0
        context_count = 0
        
        for ctx in contexts:
            if current_length + len(ctx) > max_context_length:
                break
            truncated_contexts.append(ctx)
            current_length += len(ctx)
            context_count += 1
        
        context_text = chr(10).join(truncated_contexts)
        total_results = len(contexts)
        
        # Add info about truncation
        truncation_info = f"\n\n[Hinweis: {context_count} von {total_results} relevanten Empfehlungen werden angezeigt. Die Ergebnisse sind nach Relevanz sortiert.]"
        context_text += truncation_info
    
    prompt = f"""Du beantwortest Fragen NUR auf Basis der folgenden Kontexte. Antworte auf Deutsch und füge kurze Quellenangaben an.

Kontexte:
{context_text}

Aufgabe:
1) Antworte prägnant auf die Frage.
2) Danach gib unter der Überschrift 'Originale Empfehlungstexte' die wörtlichen Empfehlungstexte der von dir genutzten Kontexte wieder – jeweils als kurzes Blockzitat in Anführungszeichen, zusammen mit Assembly (Titel, Bundesland) und Quelle (URL).

Frage: {user_q}
Antwort (mit Quellenangaben und 'Originale Empfehlungstexte'):"""
    try:
        # Some models (like GPT-5) only support default temperature
        model_params = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role":"system","content":"Du bist ein präziser Analyst. Antworte kurz, sachlich, mit Quellen aus dem Kontext."},
                {"role":"user","content": prompt}
            ]
        }
        
        # Only add temperature for models that support it
        if OPENAI_MODEL not in ["gpt-5"]:
            model_params["temperature"] = 0.1
            
        resp = client.chat.completions.create(**model_params)
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Fehler beim Abruf: {e}"
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.chat_history.append(("user", user_q))
    st.session_state.chat_history.append(("assistant", answer))

st.markdown(
    """
    <style>
    .floating-chat-btn { position: fixed; bottom: 24px; right: 24px; z-index: 1000; }
    </style>
    """,
    unsafe_allow_html=True,
)
if st.button("💬 Chat", key="open_chat_btn", help="Chat mit den Empfehlungen öffnen"):
    show_chat_dialog()
