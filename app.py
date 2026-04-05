import pickle

import streamlit as st

from entity_extractor import build_entity_graph
from enhanced_query import filtered_search
from project_paths import FAISS_CHUNKS
from query import answer_question, load_chunks
from timeline import build_timeline


st.set_page_config(layout="wide", page_title="Epstein Files AI")
st.title("Epstein Files AI")


def get_chunks():
    if FAISS_CHUNKS.exists():
        with open(FAISS_CHUNKS, "rb") as file_obj:
            return pickle.load(file_obj)
    return load_chunks()


chunks = get_chunks()

with st.sidebar:
    st.header("Filters")
    top_k = st.slider("Top results", 3, 15, 8)
    file_filter = st.text_input("Filter by file name")
    type_filter = st.selectbox("Type", ["all", "text", "image_description"])

    if not chunks:
        st.warning(
            "No indexed content found yet. Run the extraction pipeline first to create searchable data."
        )
    else:
        st.caption(f"Loaded {len(chunks):,} chunks")


tab1, tab2, tab3 = st.tabs(["Chat", "Entity Graph", "Timeline"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the files...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        results = filtered_search(
            prompt,
            k=top_k,
            file_filter=file_filter or None,
            type_filter=None if type_filter == "all" else type_filter,
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer, _ = answer_question(prompt)
            st.markdown(answer)

            if results:
                with st.expander("Sources"):
                    for item in results:
                        st.markdown(
                            f"**{item.get('file', 'Unknown')} | Page {item.get('page', '?')}**"
                        )
                        st.write(item.get("content", "")[:300])
                        image_path = item.get("image_path")
                        if image_path:
                            st.image(image_path)
            else:
                st.info("No matching sources were found in the current local index.")

        st.session_state.messages.append({"role": "assistant", "content": answer})


with tab2:
    st.subheader("Entity Relationships")
    if not chunks:
        st.info("Entity graph will appear after searchable chunks are available.")
    else:
        graph = build_entity_graph(chunks)
        entity = st.text_input("Enter a name")

        if entity:
            connections = graph.get(entity)
            if connections:
                st.write(f"Connections for **{entity}**:")
                for connection in list(sorted(connections))[:20]:
                    st.write(f"- {connection}")
            else:
                st.info("No matching entity was found in the current chunk set.")


with tab3:
    st.subheader("Timeline")
    if not chunks:
        st.info("Timeline will appear after searchable chunks are available.")
    else:
        timeline = build_timeline(chunks)
        if not timeline:
            st.info("No year-like dates were found in the current chunk set.")
        else:
            for item in timeline[:50]:
                st.markdown(f"**{item['year']}** - {item['file']} (p{item.get('page', '?')})")
                st.write(item["text"])
