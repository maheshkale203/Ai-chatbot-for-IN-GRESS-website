import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
import os
from datetime import datetime

# Language detection & translation
from deep_translator import GoogleTranslator
from langdetect import detect

# optional NLP / OpenAI
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -------------------- Config --------------------
st.set_page_config(page_title="Ingres - INGRES Groundwater Assistant", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #45a049, #4CAF50);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #ddd;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #ddd;
        padding: 10px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    .user-message {
        background: linear-gradient(135deg, #DCF8C6, #C8E6C9);
        text-align: right;
        margin-left: auto;
        color: #2E7D32;
    }
    .assistant-message {
        background: linear-gradient(135deg, #FFFFFF, #F5F5F5);
        text-align: left;
        margin-right: auto;
        color: #424242;
        border-left: 4px solid #4CAF50;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stTitle {
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
    }
    .stMarkdown h1 {
        color: #2E7D32;
    }
    .stCaption {
        color: #666;
        font-style: italic;
    }
    .stSuccess {
        background-color: #E8F5E8;
        color: #2E7D32;
        border-radius: 8px;
        padding: 10px;
    }
    .stWarning {
        background-color: #FFF3E0;
        color: #E65100;
        border-radius: 8px;
        padding: 10px;
    }
    .stError {
        background-color: #FFEBEE;
        color: #C62828;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Default dataset path (use your uploaded path)
DEFAULT_DATA_PATH = "ingress data.csv.csv"

YEAR_RE = re.compile(r"(19|20)\d{2}")
PRE_RE = re.compile(r"pre[-_ ]?monsoon", re.IGNORECASE)
POST_RE = re.compile(r"post[-_ ]?monsoon", re.IGNORECASE)

# -------------------- Utilities --------------------

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"


def translate_text(text, dest_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=dest_lang).translate(text)
    except Exception:
        return text


def safe_read(path_or_buffer):
    if path_or_buffer is None:
        return None
    try:
        if hasattr(path_or_buffer, "read"):
            return pd.read_csv(path_or_buffer, encoding="latin1", low_memory=False)
        return pd.read_csv(path_or_buffer, encoding="latin1", low_memory=False)
    except Exception:
        try:
            if hasattr(path_or_buffer, "read"):
                path_or_buffer.seek(0)
            return pd.read_excel(path_or_buffer)
        except Exception as e:
            st.sidebar.error(f"Failed to read dataset: {e}")
            return None


def get_col(df, options):
    """Return first matching column name from options that exists in df, else None."""
    for c in options:
        if c in df.columns:
            return c
    # try case-insensitive fallback
    for col in df.columns:
        for c in options:
            if c.lower() == col.lower():
                return col
    return None


def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")


def normalize_text(s):
    if s is None:
        return ""
    if pd.isna(s):
        return ""
    # remove parenthetical LGD codes, trim and lowercase
    return re.sub(r"\s*\([^\)]*\)", "", str(s)).strip().lower()

# -------------------- Column detection helper --------------------

def auto_detect_geo_columns(df):
    # possibilities (ordered)
    col_map = {
        "state": ["State_Name_With_LGD_Code", "State", "state_name", "State_Name"],
        "district": ["District_Name_With_LGD_Code", "District", "district_name", "District_Name"],
        "block": ["Block_Name_With_LGD_Code", "Block", "block_name"],
        "village": ["Village", "Village_Name", "village_name"],
        "gp": ["GP_Name_With_LGD_Code", "GP", "Gram_Panchayat"],
        "site": ["Site_Name", "Well_ID", "Well_Name", "Well"]
    }
    detected = {}
    for key, opts in col_map.items():
        detected[key] = get_col(df, opts)
    return detected

# -------------------- Convert wide to long --------------------

def detect_monsoon_columns(df):
    monsoon_map = {}
    for c in df.columns:
        low = c.lower()
        m = YEAR_RE.search(low)
        if not m:
            continue
        yr = int(m.group())
        if PRE_RE.search(low):
            monsoon_map.setdefault(yr, {})["pre"] = c
        if POST_RE.search(low):
            monsoon_map.setdefault(yr, {})["post"] = c
    return dict(sorted(monsoon_map.items()))


def melt_wide_to_long(df, monsoon_map, district_col, site_col):
    rows = []
    if not monsoon_map:
        return pd.DataFrame(rows)
    for idx, r in df.iterrows():
        district = r.get(district_col, "") if district_col and district_col in df.columns else ""
        site = r.get(site_col, f"well_{idx}") if site_col and site_col in df.columns else f"well_{idx}"
        for year, parts in monsoon_map.items():
            pre_col = parts.get("pre")
            post_col = parts.get("post")
            if pre_col and pre_col in df.columns:
                rows.append({"district": district, "well_id": site, "year": year, "season": "pre", "value_raw": r.get(pre_col)})
            if post_col and post_col in df.columns:
                rows.append({"district": district, "well_id": site, "year": year, "season": "post", "value_raw": r.get(post_col)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long["value"] = to_numeric_safe(long["value_raw"])
    long["district_norm"] = long["district"].astype(str).apply(normalize_text)
    long["well_id"] = long["well_id"].astype(str)
    return long.drop(columns=["value_raw"])

# -------------------- Simple matching helpers --------------------

def find_match(q, items):
    """Return first item string from items that is found as substring in q (case-insensitive).
    items are expected to be normalized strings (lowercase, no parenthesis).
    """
    if not q or not items:
        return None
    ql = q.lower()
    for it in items:
        if not it:
            continue
        # exact substring
        if it in ql:
            return it
        # partial match: check first 4 chars (if item longer)
        if len(it) >= 4 and it[:4] in ql:
            return it
        # token prefixes
        parts = it.split()
        for p in parts:
            if len(p) >= 4 and p[:4] in ql:
                return it
    return None


def extract_years(q):
    if not q:
        return []
    return sorted({int(m.group()) for m in YEAR_RE.finditer(q)})

# -------------------- Local NLP parser --------------------
# Note: accepts lists of known locations (normalized) and returns the parsed fields.
def nlp_parse_local(query, state_list, district_list, block_list, village_list):
    q = normalize_text(query)
    return {
        "state": find_match(q, state_list),
        "district": find_match(q, district_list),
        "block": find_match(q, block_list),
        "village": find_match(q, village_list),
        "years": extract_years(query),
        "season": "pre" if "pre" in q else "post" if "post" in q else "both",
        "intent": "trend" if any(w in q for w in ["trend","graph","plot","over time","compare"]) else "single"
    }


# -------------------- LLM + query parser (new) --------------------
def parse_user_query(query, df, district_col_geo, district_list, state_list, block_list, village_list, client=None, provider="Gemini"):
    """
    Returns a dictionary:
    {
        'intent': 'chat' | 'single' | 'trend',
        'district': str | None,
        'years': list,
        'season': 'pre'|'post'|'both'
    }
    """
    q_en = translate_text(query, dest_lang="en")

    parsed = None
    if client:
        try:
            if provider == "Gemini":
                system_prompt = (
                    "You are a JSON parser for a groundwater chatbot. "
                    "Return only JSON with fields: intent ('chat','single','trend'), district (str|null), "
                    "years (list), season ('pre','post','both'). Known districts: "
                    + ", ".join(map(str, district_list[:200]))
                )
                user_prompt = f"Question: {q_en}\nReturn only JSON."
                
                full_prompt = system_prompt + "\n" + user_prompt
                
                resp = client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=full_prompt
                )
                
                import json, re
                text = resp.text.strip()
            elif provider == "OpenAI":
                system_prompt = (
                    "You are a JSON parser for a groundwater chatbot. "
                    "Return only JSON with fields: intent ('chat','single','trend'), district (str|null), "
                    "years (list), season ('pre','post','both'). Known districts: "
                    + ", ".join(map(str, district_list[:200]))
                )
                user_prompt = f"Question: {q_en}\nReturn only JSON."
                
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                    temperature=0.0,
                    max_tokens=300
                )
                
                import json, re
                text = resp.choices[0].message.content.strip()
            
            try:
                parsed = json.loads(text)
            except:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
        except Exception:
            parsed = None

    # fallback to local NLP if LLM fails
    if not parsed:
        parsed = nlp_parse_local(q_en, state_list, district_list, block_list, village_list)

    # Ensure casual chat questions are handled
    casual = ["hi","hello","how are you","bye","thanks","name","who are you"]
    if any(c in q_en.lower() for c in casual):
        parsed['intent'] = 'chat'
        
    # Standardize empty lists/None fields
    if not parsed.get("years"):
        parsed["years"] = []
    if parsed.get("district") == "null":
        parsed["district"] = None

    return parsed, q_en


# -------------------- UI / App --------------------
# Sidebar removed - data loads automatically from default file

# Language detection is automatic

# Load dataset from default path
df_raw = None
if os.path.exists(DEFAULT_DATA_PATH):
    df_raw = safe_read(DEFAULT_DATA_PATH)

if df_raw is None:
    st.error("No dataset loaded. Please ensure 'ingress data.csv.csv' exists in the directory.")
    st.stop()

# Auto-detect geo columns and monsoon columns
geo_cols = auto_detect_geo_columns(df_raw)
monsoon_map = detect_monsoon_columns(df_raw)

# Resolve required column names (fall back to detected or common defaults)
state_col = geo_cols.get("state")
district_col_geo = geo_cols.get("district")
block_col = geo_cols.get("block")
village_col = geo_cols.get("village")
site_col = geo_cols.get("site") or geo_cols.get("gp")  # site/Well/GP fallback

# If any of essential geo columns missing, try some reasonable defaults
if district_col_geo is None:
    for c in df_raw.columns:
        if "district" in c.lower():
            district_col_geo = c
            break

if site_col is None:
    for c in df_raw.columns:
        if any(k in c.lower() for k in ["site","well","bore","site_name"]):
            site_col = c
            break

# Build distinct lists (safe)
def safe_unique_list(df, col):
    if col and col in df.columns:
        return df[col].dropna().astype(str).unique().tolist()
    return []

# Normalize lists and keep mapping normalized->original
def normalize_list_and_map(lst):
    mp = {}
    for it in lst:
        norm = normalize_text(it)
        if norm not in mp:
            mp[norm] = it
    return mp

state_map = normalize_list_and_map(safe_unique_list(df_raw, state_col))
district_map = normalize_list_and_map(safe_unique_list(df_raw, district_col_geo))
block_map = normalize_list_and_map(safe_unique_list(df_raw, block_col))
village_map = normalize_list_and_map(safe_unique_list(df_raw, village_col))

# normalized lists
state_list = list(state_map.keys())
district_list = list(district_map.keys())
block_list = list(block_map.keys())
village_list = list(village_map.keys())

# UI title & messages
st.title("💬 Ingres - INGRES Groundwater Assistant")
st.markdown("### 🌊 Welcome to Ingres, your expert assistant for INGRES groundwater data analysis!")
llm_provider = "Gemini"  # Default to Gemini

api_key = "AIzaSyBW6QAmFkgbPzaHDKbzukVHYsy8MLd86lE"  # Hardcoded API key
client = None
if GEMINI_AVAILABLE:
    try:
        client = genai.Client(api_key="AIzaSyBW6QAmFkgbPzaHDKbzukVHYsy8MLd86lE")
        st.success("Gemini connected.")
    except Exception:
        client = None
else:
    client = None


if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render previous messages
for i, m in enumerate(st.session_state["messages"]):
    if m["role"] == "user":
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f'<div class="chat-message user-message">🧑‍💬 {m["content"]}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Edit", key=f"edit_{i}", help="Edit this message"):
                st.session_state['editing'] = i
    else:
        st.markdown(f'<div class="chat-message assistant-message">🤖 {m["content"]}</div>', unsafe_allow_html=True)

# Edit functionality
if 'editing' in st.session_state:
    edit_idx = st.session_state['editing']
    current_msg = st.session_state["messages"][edit_idx]["content"]
    st.markdown("---")
    with st.form(key="edit_form"):
        st.markdown("**Edit your message:**")
        new_msg = st.text_input("Edit your message:", value=current_msg)
        submitted = st.form_submit_button("💾 Save Edit")
    
    if submitted:
        st.session_state["messages"][edit_idx]["content"] = new_msg
        # Remove messages after the edited one and re-process
        st.session_state["messages"] = st.session_state["messages"][:edit_idx+1]
        st.session_state['pending_query'] = new_msg
        del st.session_state['editing']
        st.rerun()
    
    # Cancel button outside form
    if st.button("❌ Cancel Edit"):
        del st.session_state['editing']
        st.rerun()

st.markdown("---")
query = st.chat_input("Ask your question:")

pending = st.session_state.get('pending_query')
if query or pending:
    if pending:
        q = pending
        del st.session_state['pending_query']
    else:
        q = query.strip()

    if not q:
        st.warning("Please type a question.")
        st.rerun()

    # Store user msg
    st.session_state["messages"].append({"role": "user", "content": q})
    
    with st.spinner("Processing your query..."):
        # --- Determine Target Language for Reply ---
        target_lang = detect_language(q)


        # Parse
        parsed, q_en = parse_user_query(q, df_raw, district_col_geo, district_list, state_list, block_list, village_list, client, llm_provider)
        intent = parsed.get("intent", "single")
        
        # =============================
        # CHAT INTENT (General Questions)
        # =============================
        if intent == 'chat':
            if client:
                try:
                    system_prompt = (
                        "You are Ingres, a specialized AI assistant expert in the INGRES (INDIA-Groundwater Resource Estimation System) website and groundwater data analysis. "
                        "Your primary role is to provide comprehensive information about the INGRES website and analyze water level data from CSV files. "
                        "You can respond to user queries in any language worldwide. "
                        "Answer any questions about the INGRES website with detailed, accurate information. "
                        "Analyze and provide insights from uploaded water level CSV data files. "
                        "Support multilingual queries - detect the user's language and respond in the same language. "
                        "Use your internal knowledge base about INGRES website content to answer questions. "
                        "When analyzing CSV data, provide statistical insights, trends, and visualizations when possible. "
                        "Always be helpful, accurate, and comprehensive in your responses. "
                        "For complex data analysis, use appropriate analytical tools and present findings clearly. "
                        "When provided with water level CSV data, analyze patterns, trends, seasonal variations, and anomalies. "
                        "Calculate statistical measures like mean, median, standard deviation when relevant. "
                        "Identify critical water level thresholds and alert conditions. "
                        "Compare data across different time periods, locations, or monitoring stations. "
                        "Provide actionable insights for groundwater management. "
                        "Structure responses with clear headings and bullet points. "
                        "For data analysis, include: Summary statistics, Key findings and trends, Visual descriptions or recommendations for charts, Actionable insights. "
                        "For website information, provide: Detailed explanations of INGRES features and functionality, Step-by-step guidance for using the system, Relevant links and resources when available. "
                        "Always maintain the user's preferred language throughout the response."
                    )
                    user_prompt = f"Question: {q_en}\n\nProvide a detailed response."

                    full_prompt = system_prompt + "\n\n" + user_prompt
                    resp = client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=full_prompt
                    )
                    reply_text = resp.text

                    # Translate if needed
                    if target_lang != "en":
                        reply_text = translate_text(reply_text, dest_lang=target_lang)

                except Exception as e:
                    reply_text = "I'm sorry, I encountered an error while processing your question. Please try again."
            else:
                # Fallback without LLM
                base_reply = (
                    "Hello! I am Ingres, your specialized AI assistant for the INGRES (INDIA-Groundwater Resource Estimation System) website and groundwater data analysis. "
                    "I can provide comprehensive information about INGRES, analyze water level data from CSV files, and answer questions in multiple languages. "
                    "For real-time data, I recommend checking official sources like government websites or monitoring stations. "
                    "What would you like to know about INGRES or your groundwater data?"
                )
                if target_lang != "en":
                    reply_text = translate_text(base_reply, dest_lang=target_lang)
                else:
                    reply_text = base_reply
            
            st.session_state["messages"].append(
                {"role": "assistant", "content": reply_text}
            )
            st.rerun()


        # =============================
        # District inference block
        # =============================
        if not parsed.get("district"):
            inferred = None
            v = parsed.get("village")
            if v and village_col:
                try:
                    matched = df_raw[
                        df_raw[village_col]
                        .astype(str)
                        .str.lower()
                        .str.contains(str(v).lower(), na=False)
                    ]
                    if not matched.empty and district_col_geo in matched.columns:
                        inferred = matched.iloc[0][district_col_geo]
                except Exception:
                    inferred = None

            if inferred:
                parsed["district"] = normalize_text(inferred)
            else:
                # Use target_lang for error message
                error_msg_en = "Please include a district name or a location I can infer the district from."
                error_msg = translate_text(error_msg_en, dest_lang=target_lang)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_msg}
                )
                st.rerun()

        district_parsed_norm = parsed.get("district")

        # =============================
        # Long DF Preparation
        # =============================
        long_df = melt_wide_to_long(df_raw, monsoon_map, district_col_geo, site_col)

        if long_df.empty:
            error_msg_en = "No monsoon data columns (e.g., Pre-Monsoon 2020) were found in the dataset."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
            st.rerun()

        # District filter
        if district_parsed_norm:
            df_filtered = long_df[
                long_df["district_norm"].str.contains(district_parsed_norm, na=False)
            ]
        else:
            df_filtered = pd.DataFrame() # Should not happen due to check above, but safe.

        # Approx match fallback
        if df_filtered.empty and district_parsed_norm:
            tokens = re.findall(r"[A-Za-z]{3,}", str(district_parsed_norm or ""))
            for t in tokens:
                cand = long_df[long_df["district_norm"].str.contains(t, na=False)]
                if not cand.empty:
                    df_filtered = cand
                    break

        if df_filtered.empty:
            error_msg_en = f"No data found for district '{parsed.get('district')}'."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )
            st.rerun()

        # =============================
        # Trend Graph Block
        # =============================
        if intent == "trend":
            display_name = district_map.get(
                district_parsed_norm, parsed.get("district")
            )

            response_msg_en = f"Here is the trend for *{display_name}*."
            response_msg = translate_text(response_msg_en, dest_lang=target_lang)
            
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_msg}
            )
            
            pre_title_en = f"Pre-monsoon Trend — {display_name}"
            post_title_en = f"Post-monsoon Trend — {display_name}"
            
            # Note: We don't translate chart titles as Altair/Streamlit don't support multi-language titles easily,
            # but we can translate the base message.

            df_pre = df_filtered[df_filtered["season"] == "pre"]
            df_post = df_filtered[df_filtered["season"] == "post"]

            if not df_pre.empty:
                st.altair_chart(
                    alt.Chart(df_pre)
                    .mark_line(point=True)
                    .encode(x=alt.X("year:O", title="Year"), y=alt.Y("value:Q", title="Groundwater Level"), color="well_id:N")
                    .properties(title=pre_title_en)
                    .interactive()
                )

            if not df_post.empty:
                st.altair_chart(
                    alt.Chart(df_post)
                    .mark_line(point=True)
                    .encode(x=alt.X("year:O", title="Year"), y=alt.Y("value:Q", title="Groundwater Level"), color="well_id:N")
                    .properties(title=post_title_en)
                    .interactive()
                )

            st.rerun()

        # =============================
        # Single Year / Value Block
        # =============================
        
        years_parsed = parsed.get("years", [])

        target_year = (
            years_parsed[0]
            if years_parsed
            else max(df_filtered["year"].unique())
            if not df_filtered.empty
            else None
        )

        if not target_year:
            error_msg_en = "Please include a year in your query, or try a 'trend' question."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
            st.rerun()

        df_year = df_filtered[df_filtered["year"] == target_year]

        if df_year.empty:
            error_msg_en = f"No data for {district_map.get(district_parsed_norm)} in {target_year}."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )
            st.rerun()

        # =============================
        # Pivot + averages
        # =============================
        pivot = (
            df_year.pivot_table(
                index="well_id",
                columns="season",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
        )

        pre_avg = pivot["pre"].mean() if "pre" in pivot.columns else np.nan
        post_avg = pivot["post"].mean() if "post" in pivot.columns else np.nan

        data_context = f"District: {district_map.get(district_parsed_norm)}, Year: {target_year}, Pre-Monsoon Avg: {pre_avg:.2f}, Post-Monsoon Avg: {post_avg:.2f}."

        # =============================
        # LLM ANSWER
        # =============================
        if client and (llm_provider == "Gemini" or llm_provider == "OpenAI"):
            try:
                system_prompt = (
                    "You are Ingres, a specialized AI assistant expert in the INGRES (INDIA-Groundwater Resource Estimation System) website and groundwater data analysis. "
                    "Your primary role is to provide comprehensive information about the INGRES website and analyze water level data from CSV files. "
                    "You can respond to user queries in any language worldwide. "
                    "Answer any questions about the INGRES website with detailed, accurate information. "
                    "Analyze and provide insights from uploaded water level CSV data files. "
                    "Support multilingual queries - detect the user's language and respond in the same language. "
                    "Use your internal knowledge base about INGRES website content to answer questions. "
                    "When analyzing CSV data, provide statistical insights, trends, and visualizations when possible. "
                    "Always be helpful, accurate, and comprehensive in your responses. "
                    "For complex data analysis, use appropriate analytical tools and present findings clearly. "
                    "When provided with water level CSV data, analyze patterns, trends, seasonal variations, and anomalies. "
                    "Calculate statistical measures like mean, median, standard deviation when relevant. "
                    "Identify critical water level thresholds and alert conditions. "
                    "Compare data across different time periods, locations, or monitoring stations. "
                    "Provide actionable insights for groundwater management. "
                    "Structure responses with clear headings and bullet points. "
                    "For data analysis, structure the response as follows: "
                    "- Overview: Provide a brief overview of the groundwater levels for the specified location and year. "
                    "- Key Points: "
                    "  - Pre-Monsoon [Year]: Describe the pre-monsoon levels, trends, and any notable regions. "
                    "  - Post-Monsoon [Year]: Describe the post-monsoon levels, improvements or declines. "
                    "  - Statewide Trends (if applicable): Overall trends across the area. "
                    "  - Long-Term Change: Comparison to previous years or decadal averages. "
                    "- Data Sources: Mention sources like Central Ground Water Board (CGWB), Maharashtra Groundwater Survey and Development Agency (GSDA), and the uploaded dataset. "
                    "- Example Table: Provide a table with approximate averages for different regions if applicable, using mbgl (meters below ground level). "
                    "- Summary: Summarize key insights for different regions. "
                    "- End with an offer for more specific data. "
                    "For website information, provide: Detailed explanations of INGRES features and functionality, Step-by-step guidance for using the system, Relevant links and resources when available. "
                    "Always maintain the user's preferred language throughout the response."
                )
                user_prompt = (
                    f"Original Question (translated to English): {q_en}\n\nData Context: {data_context}\n\n"
                    "Provide a detailed, helpful response as a groundwater expert."
                )

                full_prompt = system_prompt + "\n\n" + user_prompt
                resp = client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=full_prompt
                )
                reply_text = resp.text

            except Exception as e:
                # Fallback on LLM error
                reply_text = f"An error occurred while generating a detailed response. The core data is: {data_context}"

        else:
            # Fallback when no API key is available - provide detailed analysis like Gemini would
            district_name = district_map.get(district_parsed_norm, district_parsed_norm)
            reply_text = (
                f"{district_name} Groundwater Level in {target_year}\n\n"
                f"Overview: In {target_year}, {district_name} showed groundwater levels based on the available data from monitoring wells. "
                f"The levels reflect seasonal variations due to monsoon patterns and local extraction rates.\n\n"
                f"Key Points:\n"
                f"Pre-Monsoon {target_year}:\n"
                f"- Average groundwater level: {pre_avg:.2f} meters below ground level (mbgl)\n"
                f"- Represents the water table before the rainy season, typically the driest period.\n"
                f"- Levels may indicate stress from previous dry periods or over-extraction.\n\n"
                f"Post-Monsoon {target_year}:\n"
                f"- Average groundwater level: {post_avg:.2f} mbgl\n"
                f"- Shows levels after rainfall recharge.\n"
                f"- Improvement suggests good monsoon rainfall and recharge.\n\n"
                f"Statewide Trends:\n"
                f"- The difference between post and pre-monsoon levels indicates seasonal recharge: "
                f"{'positive recharge' if post_avg < pre_avg else 'limited recharge or continued decline'}.\n\n"
                f"Long-Term Change:\n"
                f"- Comparison to previous years not available from current dataset. Monitor over multiple years for trends.\n\n"
                f"Data Sources:\n"
                f"- Uploaded groundwater dataset\n"
                f"- Central Ground Water Board (CGWB)\n"
                f"- Maharashtra Groundwater Survey and Development Agency (GSDA)\n\n"
                f"Example Table (Approximate Averages):\n"
                f"| District | Pre-Monsoon {target_year} (mbgl) | Post-Monsoon {target_year} (mbgl) |\n"
                f"|----------|-----------------------------------|------------------------------------|\n"
                f"| {district_name} | {pre_avg:.2f} | {post_avg:.2f} |\n\n"
                f"Summary:\n"
                f"- {district_name}: {'Improved levels post-monsoon' if post_avg < pre_avg else 'Continued stress with limited improvement'}.\n"
                f"- For sustainable management, aim for post-monsoon levels higher than pre-monsoon.\n\n"
                f"If you need data for a specific district or block, let me know!"
            )

        # Final Translation (if not English)
        if target_lang != "en":
            reply_text = translate_text(reply_text, dest_lang=target_lang)

        st.session_state["messages"].append(
            {"role": "assistant", "content": reply_text}
        )

        # Show pivot
        if not pivot.empty:
            st.caption(f"Raw data for {district_map.get(district_parsed_norm)} in {target_year}:")
            st.dataframe(pivot)

        st.rerun()

# =============================
# Clear Button
# =============================
if st.button("🗑️ Clear chat"):
    st.session_state["messages"] = []
    st.rerun()

st.caption(f"App last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")