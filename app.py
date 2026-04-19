import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datetime
from collections import Counter
import streamlit as st

from enhanced_query import filtered_search
from query import answer_question, search

try:
    from entity_extractor import build_entity_graph
except ImportError:
    def build_entity_graph(results):
        st.warning("entity_extractor.py not found — Entity Graph unavailable.")
        return {}

try:
    from timeline import build_timeline
except ImportError:
    def build_timeline(results):
        st.warning("timeline.py not found — Timeline unavailable.")
        return []


st.set_page_config(
    layout="wide",
    page_title="DOJ Disclosure Archive",
    page_icon=None,
)

st.markdown("""
<style>

/* --- GLOBAL --- */
body, .stApp {
    background-color: #F5F7FA;
    color: #2C3E50;
    font-family: 'Georgia', serif;
}

/* --- SIDEBAR --- */
[data-testid="stSidebar"] {
    background-color: #2C3E50 !important;
    border-right: 1px solid #3d5166;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #DCE3EA !important; }

/* Radio */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    appearance: none !important; -webkit-appearance: none !important;
    width: 14px !important; height: 14px !important;
    border-radius: 50% !important; border: 2px solid #9BAFC2 !important;
    background-color: transparent !important; display: inline-block !important;
    flex-shrink: 0 !important; margin-right: 10px !important;
    cursor: pointer !important; transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"]:checked {
    background-color: #FFFFFF !important; border-color: #FFFFFF !important;
    box-shadow: inset 0 0 0 3px #2C3E50 !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important; align-items: center !important;
    padding: 5px 4px !important; margin: 1px 0 !important;
    background: transparent !important; border: none !important; cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span { color: #9BAFC2 !important; font-size: 14px !important; background: transparent !important; }
[data-testid="stSidebar"] .stRadio label:has(input:checked) p,
[data-testid="stSidebar"] .stRadio label:has(input:checked) span { color: #FFFFFF !important; font-weight: 600 !important; }

/* Kill red focus rings */
input, textarea, [data-baseweb="input"], [data-baseweb="textarea"] { outline: none !important; box-shadow: none !important; }
*:focus, *:focus-within, *:focus-visible { outline-color: #4A6FA5 !important; border-color: #4A6FA5 !important; }
[data-baseweb="base-input"]:focus-within, div[data-focused="true"] {
    border-color: #4A6FA5 !important; box-shadow: 0 0 0 2px rgba(74,111,165,0.2) !important;
}

.stTextInput input {
    background: #FFFFFF !important; border: 1px solid #DCE3EA !important;
    border-radius: 4px !important; color: #2C3E50 !important; font-family: 'Georgia', serif !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #3d5166 !important; border: 1px solid #4A6FA5 !important; color: #DCE3EA !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #3d5166 !important; border-color: #4A6FA5 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { background-color: transparent !important; color: #DCE3EA !important; }

/* --- BUTTONS --- */
.stButton > button {
    background: #4A6FA5 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 4px !important; font-family: 'Georgia', serif !important;
    padding: 6px 14px !important; font-size: 13px !important;
}
.stButton > button:hover { background: #2C3E50 !important; }

/* --- CHAT MESSAGES --- */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important; border: 1px solid #DCE3EA !important;
    border-radius: 6px !important; font-size: 15px !important;
}
[data-testid="chatAvatarIcon-assistant"] { background-color: #4A6FA5 !important; color: #FFFFFF !important; }
[data-testid="chatAvatarIcon-user"] { background-color: #DCE3EA !important; color: #2C3E50 !important; }

/* --- SOURCE CARDS --- */
.source-card {
    background: #FFFFFF; border: 1px solid #DCE3EA; border-left: 4px solid #4A6FA5;
    border-radius: 4px; padding: 10px; font-family: 'Georgia', serif; margin-bottom: 8px;
}
.redacted-badge { background: #2C3E50; color: #DCE3EA; font-size: 9px; padding: 2px 6px; border-radius: 2px; }

/* --- SECTION HEADERS --- */
.section-header {
    font-family: 'Georgia', serif; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4A6FA5; border-bottom: 2px solid #DCE3EA;
    padding-bottom: 6px; margin-bottom: 12px;
}
.context-indicator { font-size: 11px; color: #4A6FA5; font-family: 'Georgia', serif; font-style: italic; }
.stProgress > div > div { background-color: #4A6FA5 !important; }

/* --- DOWNLOAD BUTTON --- */
.stDownloadButton > button {
    background: #F5F7FA !important; color: #4A6FA5 !important;
    border: 1px solid #4A6FA5 !important; border-radius: 4px !important; font-size: 12px !important;
}
.stDownloadButton > button:hover { background: #4A6FA5 !important; color: #FFFFFF !important; }

/* === STREAMLIT NATIVE CHAT INPUT — styled to match Claude/ChatGPT === */

/* The bar Streamlit renders: pin it, give it a solid background */
[data-testid="stBottom"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 245px !important;
    right: 0 !important;
    z-index: 9999 !important;
    background: #F5F7FA !important;
    border-top: 1px solid #DCE3EA !important;
    padding: 12px 24px 16px !important;
    box-sizing: border-box !important;
}
[data-testid="stBottom"] > div {
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* ONE box: style only the inner baseweb wrapper, kill borders everywhere else */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
/* The actual visible pill */
[data-testid="stChatInput"] > div {
    background: #FFFFFF !important;
    border-radius: 32px !important;
    border: 1.5px solid #DCE3EA !important;
    padding: 6px 8px 6px 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07) !important;
    display: flex !important;
    align-items: center !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #4A6FA5 !important;
    box-shadow: 0 4px 18px rgba(74,111,165,0.13) !important;
}
/* Textarea: no extra borders */
[data-testid="stChatInput"] textarea {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-family: Georgia, serif !important;
    font-size: 15px !important;
    color: #2C3E50 !important;
    padding: 8px 4px !important;
    resize: none !important;
}
/* Send button: dark circle + white arrow */
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button {
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    background-color: #1e293b !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
    padding: 0 !important;
    transition: background-color 0.2s !important;
}
[data-testid="stChatInputSubmitButton"]:hover,
[data-testid="stChatInput"] button:hover {
    background-color: #4A6FA5 !important;
}
[data-testid="stChatInputSubmitButton"] svg,
[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
    width: 16px !important; height: 16px !important;
}
[data-testid="stChatInputSubmitButton"] svg path,
[data-testid="stChatInput"] button svg path {
    fill: #FFFFFF !important;
    stroke: none !important;
}

/* Push content up so last message isn't hidden behind bar */
section.main > div { padding-bottom: 110px !important; }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def redaction_level(text):
    count = text.count("[REDACTED]") + text.count("█") + text.count("XXXX")
    if count >= 4:
        return "heavily"
    elif count >= 1:
        return "partial"
    return None


def export_markdown(query, answer, results):
    lines = [
        "# DOJ Disclosure Archive — Export\n",
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        f"## Query\n\n{query}\n\n",
        f"## Answer\n\n{answer}\n\n",
        "## Sources\n\n",
    ]
    for i, r in enumerate(results[:10], 1):
        lines.append(f"**{i}. {r.get('file', 'Unknown')} — Page {r.get('page', '?')}**\n\n")
        lines.append(f"> {r.get('content', '')[:300]}\n\n")
    return "".join(lines)


def _remove_fixed_chat_bar():
    """No-op: the bar is rendered inline by Chat view only; other views simply don't render it."""
    pass


SUGGESTED_FOLLOWUPS = {
    "arrest":  ["What charges were filed?", "Who else was named?", "What happened after the arrest?"],
    "epstein": ["Who were his known associates?", "What properties did he own?", "What court filings exist?"],
    "witness": ["What did the witness testify?", "Were there corroborating witnesses?", "What was the outcome?"],
    "payment": ["Who received payments?", "What were the amounts?", "Were charges filed?"],
    "flight":  ["Who was on the flight logs?", "Which destinations appear most?", "What dates are documented?"],
    "maxwell": ["What was Maxwell's role?", "What charges did Maxwell face?", "What documents mention Maxwell?"],
}

def get_suggestions(query, answer):
    combined = (query + " " + answer).lower()
    for keyword, chips in SUGGESTED_FOLLOWUPS.items():
        if keyword in combined:
            return chips
    return ["What documents relate to this?", "Who else is mentioned?", "Show me the timeline for this."]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# FIX: Read pending_file_filter BEFORE rendering the widget so it actually
# takes effect. Previously it was read after widget creation and silently
# discarded because Streamlit had already captured the widget value.
_pending_file_filter = st.session_state.pop("pending_file_filter", None)

with st.sidebar:
    st.markdown(
        '<div style="padding:16px 0 8px;">'
        '<span style="font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#9BAFC2;">DOJ Disclosure</span><br>'
        '<span style="font-size:20px;color:#FFFFFF;font-family:Georgia,serif;font-weight:bold;">Epstein Archive</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:#3d5166;margin:8px 0;">', unsafe_allow_html=True)

    view = st.radio(
        "nav",
        ["Chat", "Entity Graph", "Timeline", "Heatmap"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:#3d5166;margin:8px 0;">', unsafe_allow_html=True)
    st.markdown('<span style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#9BAFC2;">Filters</span>', unsafe_allow_html=True)
    st.markdown(
    '<span style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#9BAFC2;">Analysis Mode</span>',
    unsafe_allow_html=True
    )

    mode = st.radio(
        "Analysis Mode",
        ["Quick", "Balanced", "Deep"],
        help="""
    Choose how deeply the system analyzes documents:

    • Quick → Fast answers using fewer documents (saves tokens for more questions)
    • Balanced → Best mix of speed and accuracy (recommended)
    • Deep → More thorough analysis using more documents (slower, uses more tokens)

    Tip: Using Quick or Balanced helps conserve tokens so you can ask more questions.
    """
    )
    if mode == "Quick":
        top_k = 6
    elif mode == "Balanced":
        top_k = 8
    else:
        top_k = 12

    # FIX: supply the pending value as the widget's default so it renders
    # with the correct pre-filled text rather than being ignored.
    file_filter = st.text_input(
        "File name",
        value=_pending_file_filter or "",
        placeholder="e.g. SDNY_2019",
    )
    type_filter = st.selectbox("Type", ["all", "text", "image_description"])


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

if view == "Chat":
    # Page-level title — sits at top above everything, aligned with Deploy button
    st.markdown(
        '<h1 style="font-family:Georgia,serif;font-size:32px;color:#2C3E50;'
        'font-weight:bold;margin:0 0 24px 0;padding:0;">Epstein Archive</h1>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    with col1:
        st.markdown('<div class="section-header">Case Inquiry</div>', unsafe_allow_html=True)

        if st.session_state.last_results:
            top_file = st.session_state.last_results[0].get("file", "")
            if top_file:
                st.markdown(
                    f'<span class="context-indicator">⌕ Following context: {top_file}</span>',
                    unsafe_allow_html=True,
                )

        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("chips"):
                    chip_cols = st.columns(len(message["chips"]))
                    for j, chip in enumerate(message["chips"]):
                        with chip_cols[j]:
                            if st.button(chip, key=f"chip_{i}_{j}"):
                                st.session_state.pending_prompt = chip
                                st.rerun()

        # FIX: consume pending_prompt BEFORE chat_input so that Streamlit's
        # widget system sees the resolved prompt value on the same render pass.
        # Previously, pending_prompt was popped after chat_input, meaning the
        # first rerun would render chat_input with an empty value and then
        # overwrite `prompt` — but only if the user hadn't typed anything,
        # creating a subtle race. Popping first guarantees correct ordering.
        prompt = st.session_state.pop("pending_prompt", None)
        typed  = st.chat_input("Enter your inquiry...")
        if typed:
            prompt = typed

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            history_so_far = st.session_state.messages[:-1]

            with st.chat_message("assistant"):
                with st.spinner("Searching archive..."):
                    # Single search: answer_question does its own reranked search
                    # internally. We also run filtered_search with the sidebar
                    # filters so the Evidence Sources panel respects them.
                    answer, answer_results = answer_question(prompt, history=history_so_far)
                    results = filtered_search(
                        prompt,
                        k=top_k,
                        file_filter=file_filter or None,
                        type_filter=None if type_filter == "all" else type_filter,
                    ) or answer_results  # fall back to answer_results if filter returns nothing

                st.markdown(answer)

                dl_col, _ = st.columns([1, 3])
                with dl_col:
                    st.download_button(
                        "↓ Export .md",
                        data=export_markdown(prompt, answer, results),
                        file_name=f"archive_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key=f"export_{len(st.session_state.messages)}",
                    )

                chips = get_suggestions(prompt, answer)
                chip_cols = st.columns(len(chips))
                for j, chip in enumerate(chips):
                    with chip_cols[j]:
                        if st.button(chip, key=f"livechip_{len(st.session_state.messages)}_{j}"):
                            st.session_state.pending_prompt = chip
                            st.rerun()

            st.session_state.last_results = results
            st.session_state.last_query   = prompt
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "chips":   chips,
            })

    # Right panel
    with col2:
        st.markdown('<div class="section-header">Evidence Sources</div>', unsafe_allow_html=True)
        results = st.session_state.get("last_results", [])
        if results:
            for i, item in enumerate(results[:10]):
                content = item.get("content", "")
                rlevel  = redaction_level(content)

                badge = ""
                if rlevel == "heavily":
                    badge = '<span class="redacted-badge heavily-redacted">HEAVILY REDACTED</span>'
                elif rlevel == "partial":
                    badge = '<span class="redacted-badge">PARTIAL REDACTION</span>'

                confidence = max(10, 100 - i * 9)
                bar = "█" * (confidence // 10) + "░" * (10 - confidence // 10)

                st.markdown(f"""<div class="source-card">
<div style="font-size:9px;color:#7f8c8d;margin-bottom:3px;">#{i+1} · {confidence}% relevance</div>
<div style="font-size:11px;color:#2C3E50;font-weight:bold;margin-bottom:1px;">{item.get('file','Unknown')}{badge}</div>
<div style="font-size:10px;color:#7f8c8d;margin-bottom:5px;">Page {item.get('page','?')} · {item.get('type','text')}</div>
<div style="font-size:10px;color:#4A6FA5;letter-spacing:0.04em;margin-bottom:5px;">{bar}</div>
<div style="font-size:11px;color:#2C3E50;line-height:1.5;">{content[:180]}{'…' if len(content) > 180 else ''}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="font-size:12px;color:#7f8c8d;font-family:Georgia,serif;margin-top:8px;">Awaiting inquiry…</p>',
                unsafe_allow_html=True,
            )

    # Native st.chat_input renders below — styled via CSS above


# ---------------------------------------------------------------------------
# Entity Graph
# ---------------------------------------------------------------------------

elif view == "Entity Graph":
    _remove_fixed_chat_bar()
    st.markdown('<div class="section-header">Entity Network</div>', unsafe_allow_html=True)

    if st.button("Build Entity Graph"):
        with st.spinner("Fetching documents from Pinecone..."):
            sample_results = search("epstein maxwell flight money payment", k=200)
        with st.spinner("Extracting entities..."):
            graph = build_entity_graph(sample_results)
        st.session_state["entity_graph"] = graph

    if "entity_graph" in st.session_state:
        graph  = st.session_state["entity_graph"]
        entity = st.text_input("Search entity", placeholder="Enter a name or organization...")
        if entity:
            connections = graph.get(entity)
            if connections:
                sorted_conns = list(sorted(connections))[:20]
                st.markdown(f"**{len(sorted_conns)} connections** found for: *{entity}*")
                for conn in sorted_conns:
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(
                            f'<span style="font-family:Georgia,serif;font-size:13px;color:#2C3E50;">— {conn}</span>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button("Inquire", key=f"ent_{conn}"):
                            st.session_state.pending_prompt = f"What do the documents say about {conn}?"
                            st.rerun()
            else:
                st.info("No entity found. Try a different spelling.")


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

elif view == "Timeline":
    _remove_fixed_chat_bar()
    st.markdown('<div class="section-header">Chronological Record</div>', unsafe_allow_html=True)

    if st.button("Build Timeline"):
        with st.spinner("Fetching documents from Pinecone..."):
            sample_results = search("date year meeting flight payment arrest", k=200)
        with st.spinner("Parsing dates..."):
            timeline = build_timeline(sample_results)
        st.session_state["timeline"] = timeline

    if "timeline" in st.session_state:
        timeline = st.session_state["timeline"]
        if not timeline:
            st.info("No dates found in the archive.")
        else:
            year_filter = st.text_input("Filter by year", placeholder="e.g. 2019")
            prev_year   = None
            count       = 0
            for item in timeline:
                yr = str(item.get("year", ""))
                if year_filter and year_filter not in yr:
                    continue
                if count >= 100:
                    st.caption("Showing first 100 entries.")
                    break
                if yr != prev_year:
                    st.markdown(
                        f'<div style="font-family:Georgia,serif;font-size:11px;color:#4A6FA5;'
                        f'letter-spacing:0.1em;margin-top:16px;margin-bottom:4px;'
                        f'border-bottom:1px solid #DCE3EA;padding-bottom:4px;">── {yr} ──</div>',
                        unsafe_allow_html=True,
                    )
                    prev_year = yr
                st.markdown(f"""<div class="source-card">
<div style="font-size:10px;color:#7f8c8d;">{item.get('file','?')} · p{item.get('page','?')}</div>
<div style="font-size:13px;margin-top:4px;line-height:1.5;color:#2C3E50;">{item.get('text','')[:300]}</div>
</div>""", unsafe_allow_html=True)
                count += 1


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

elif view == "Heatmap":
    _remove_fixed_chat_bar()
    st.markdown('<div class="section-header">Reference Heatmap</div>', unsafe_allow_html=True)
    st.caption("Documents ranked by reference frequency in your last query.")

    results = st.session_state.get("last_results", [])
    if not results:
        st.info("Ask questions in Chat first — the heatmap will show which documents are referenced most.")
    else:
        doc_counts = Counter(r.get("file", "Unknown") for r in results)
        max_count  = max(doc_counts.values())

        st.markdown(f"**{len(doc_counts)} documents** referenced in last query")
        st.markdown('<hr style="border-color:#DCE3EA;margin:8px 0;">', unsafe_allow_html=True)

        for fname, count in doc_counts.most_common(30):
            intensity = count / max_count
            bar_len   = int(intensity * 28)
            bar       = "█" * bar_len + "░" * (28 - bar_len)
            pct       = int(intensity * 100)

            st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:7px;font-family:Georgia,serif;font-size:11px;">
  <div style="width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#2C3E50;" title="{fname}">{fname}</div>
  <div style="color:#4A6FA5;">{bar}</div>
  <div style="color:#7f8c8d;min-width:28px;">{pct}%</div>
</div>""", unsafe_allow_html=True)