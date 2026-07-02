import streamlit as st
import requests
import uuid

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHL Assessment Finder",
    page_icon="🎯",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header */
.header-container {
    text-align: center;
    padding: 2rem 0 1rem;
}
.header-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
}

/* Status badge */
.status-badge {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #34d399;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

/* Chat messages */
.chat-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.2rem;
    margin: 0.5rem 0;
    margin-left: 15%;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    font-size: 0.95rem;
}
.chat-assistant {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.2rem;
    margin: 0.5rem 0;
    margin-right: 15%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    font-size: 0.95rem;
    backdrop-filter: blur(10px);
}
.chat-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    opacity: 0.7;
}
.chat-clarify {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: #fde68a;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.2rem;
    margin: 0.5rem 0;
    margin-right: 15%;
    font-size: 0.95rem;
}

/* Divider */
.chat-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1rem 0;
}

/* Input area */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 0.7rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
}

/* Send button */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.8rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
}

/* Clear button */
.clear-btn > button {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}
.clear-btn > button:hover {
    background: rgba(239, 68, 68, 0.25) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #a78bfa !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/chat"

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("API Endpoint", value=API_URL)
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "Hiring a Java developer who works with stakeholders",
        "I need a cognitive ability test for graduates",
        "Remote adaptive test under 30 minutes for managers",
        "Personality assessment for sales executives",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex

    st.markdown("---")
    st.markdown("### 📊 Session Info")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    st.markdown(f"**Session ID:** `{st.session_state.get('session_id', 'N/A')[:8]}...`")
    st.markdown(
        "<span class='status-badge'>🟢 Connected</span>" if True else "",
        unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🎯 SHL Assessment Finder</div>
    <div class="header-subtitle">AI-powered assessment recommendation engine</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='chat-divider'>", unsafe_allow_html=True)

# ── Chat History ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-user">
            <div class="chat-label">YOU</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    elif msg["role"] == "clarification":
        st.markdown(f"""
        <div class="chat-clarify">
            <div class="chat-label">🤔 CLARIFICATION NEEDED</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-assistant">
            <div class="chat-label">🎯 SHL ASSISTANT</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

# ── Input Area ─────────────────────────────────────────────────────────────────
st.markdown("<hr class='chat-divider'>", unsafe_allow_html=True)

prefill_value = st.session_state.pop("prefill", "")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "query",
        value=prefill_value,
        placeholder="Ask about SHL assessments...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send = st.button("Send 🚀")

col_clear, _ = st.columns([1, 4])
with col_clear:
    with st.container():
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Send Request ───────────────────────────────────────────────────────────────
if send and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    # Build the messages array for the stateless API
    # Map "clarification" role back to "assistant" for the API
    api_messages = []
    for m in st.session_state.messages:
        role = "assistant" if m["role"] == "clarification" else m["role"]
        api_messages.append({"role": role, "content": m["content"]})

    with st.spinner("Searching catalog..."):
        try:
            resp = requests.post(
                api_url,
                json={"messages": api_messages},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            # Use the API schema fields: 'reply', 'recommendations', 'end_of_conversation'
            response_text = data.get("reply", "No response.")
            end_of_conversation = data.get("end_of_conversation", True)
            recommendations = data.get("recommendations", [])
            
            # If the conversation is not over and there are no recommendations, it's a clarification prompt.
            needs_clarification = (not end_of_conversation) and (len(recommendations) == 0)

            role = "clarification" if needs_clarification else "assistant"
            st.session_state.messages.append({"role": role, "content": response_text})

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The model may be taking too long.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    st.rerun()

