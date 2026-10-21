import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import index_uploaded_files, answer_question, get_retriever, build_generator
from evaluation import load_jsonl_questions, evaluate_retrieval

load_dotenv()
st.set_page_config(page_title="NeuroRAG-FT", page_icon="📚", layout="wide")
st.title("NeuroRAG-FT")
st.caption("Research-paper assistant with retrieval-grounded answers and inspectable sources.")

if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("Configuration")
    st.info("Default mode is MOCK. It verifies UI flow but does not perform language-model inference.")
    top_k = st.slider("Retrieved passages (top-k)", 1, 10, 4)
    backend = st.selectbox("Generator backend", ["mock", "ollama"])
    model = st.text_input("Ollama model", value="llama3.1")
    st.caption("For Ollama mode, start Ollama locally and pull the selected model.")

tab_docs, tab_ask, tab_eval = st.tabs(["Documents", "Ask", "Evaluation"])

with tab_docs:
    uploads = st.file_uploader(
        "Upload research papers or notes", type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    if st.button("Index uploaded documents", type="primary", disabled=not uploads):
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            try:
                summary = index_uploaded_files(uploads)
                st.session_state.indexed = True
                st.success(f"Indexed {summary['chunks']} chunks from {summary['files']} file(s).")
                st.json(summary)
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
    st.warning("Use documents you are permitted to process. Scanned PDFs may require OCR.")

with tab_ask:
    question = st.text_area("Question", placeholder="What methods and limitations are described in the uploaded papers?")
    if st.button("Answer question", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                retriever = get_retriever()
                generator = build_generator(backend, model)
                result = answer_question(question.strip(), retriever, generator, k=top_k)
                st.subheader("Answer")
                st.write(result["answer"])
                if result.get("mock"):
                    st.warning("Mock output: no real model inference was performed.")
                st.subheader("Retrieved evidence")
                if not result["sources"]:
                    st.info("No evidence retrieved. The system abstained.")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"[S{i}] {source.get('source', 'unknown')} · page {source.get('page', '?')}"):
                        st.write(source["text"])
                        st.caption(f"Chunk ID: {source['id']} | Distance: {source.get('distance', 'n/a')}")
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

with tab_eval:
    st.markdown("Upload a JSONL file with `question` and `relevant_chunk_ids` fields to evaluate retrieval.")
    eval_file = st.file_uploader("Evaluation set (.jsonl)", type=["jsonl"], key="eval_upload")
    if eval_file and st.button("Evaluate retrieval"):
        try:
            questions = load_jsonl_questions(eval_file.getvalue().decode("utf-8"))
            metrics = evaluate_retrieval(questions, get_retriever(), k=top_k)
            st.json(metrics)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
