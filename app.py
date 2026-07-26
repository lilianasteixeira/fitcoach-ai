"""
FitCoach AI Streamlit interface: chat with the fitness/nutrition assistant,
shows the sources used in the answer, and collects user feedback
(thumbs up/down), stored in Postgres for the monitoring dashboard.
"""

import streamlit as st

from db.monitoring_db import log_conversation, save_feedback
from rag.pipeline import answer_question

st.set_page_config(page_title="FitCoach AI", page_icon="💪", layout="centered")

st.title("💪 FitCoach AI")
st.caption("RAG-powered fitness & nutrition assistant — LLM Zoomcamp final project")

with st.sidebar:
    st.header("Settings")
    search_strategy = st.selectbox("Retrieval strategy", ["hybrid", "vector", "text"], index=0)
    prompt_template = st.selectbox("Prompt template", ["detailed", "concise"], index=0)
    top_k = st.slider("Number of retrieved documents", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        "⚠️ This assistant does not replace personalized medical or "
        "nutritional advice."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['topic']}** — {s['question']} (score: {s['score']:.3f})")
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("👍", key=f"up_{msg['conversation_id']}"):
                    save_feedback(msg["conversation_id"], 1)
                    st.toast("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"down_{msg['conversation_id']}"):
                    save_feedback(msg["conversation_id"], -1)
                    st.toast("Thanks, we'll improve!")

question = st.chat_input("Ask me something about training or nutrition...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(
                question,
                search_strategy=search_strategy,
                prompt_template=prompt_template,
                top_k=top_k,
            )
            main_topic = result["sources"][0]["topic"] if result["sources"] else None
            conversation_id = log_conversation(
                question=result["question"],
                answer=result["answer"],
                model=result["model"],
                search_strategy=result["search_strategy"],
                prompt_template=result["prompt_template"],
                latency_seconds=result["latency_seconds"],
                topic=main_topic,
            )
        st.markdown(result["answer"])
        st.caption(f"⏱️ {result['latency_seconds']}s · model: {result['model']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "conversation_id": conversation_id,
        }
    )
    st.rerun()
