import os
import re
from io import BytesIO

import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Engineering Manual Search", page_icon="🛞", layout="wide")

st.title("🛞 Engineering Manual Search")
st.caption("Upload a marine equipment manual and ask questions with page citations.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Pulled automatically from Streamlit Cloud secrets if set there.",
    )
    model = st.text_input("Claude model", value="claude-sonnet-5")
    top_k = st.slider("Pages/chunks to retrieve", 2, 8, 5)
    st.divider()
    st.info(
        "Prototype only. Always verify safety-critical maintenance steps "
        "against the original manual and vessel procedures."
    )


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes: bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def chunk_pages(pages, max_chars=3500, overlap=350):
    chunks = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk = text[start:end]
            chunks.append(
                {
                    "page": item["page"],
                    "text": chunk,
                }
            )
            if end == len(text):
                break
            start = max(0, end - overlap)
    return chunks


def retrieve(question, chunks, k):
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
    )
    matrix = vectorizer.fit_transform(corpus + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:k]
    return [
        {
            **chunks[i],
            "score": float(scores[i]),
        }
        for i in ranked
    ]


def ask_claude(question, evidence, key, model_name):
    client = Anthropic(api_key=key)

    context_blocks = []
    for idx, item in enumerate(evidence, start=1):
        context_blocks.append(
            f"[SOURCE {idx} — PDF PAGE {item['page']}]\n{item['text']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = """You are a careful marine technical manual assistant.
Use only the supplied manual excerpts. Do not invent procedures, specifications,
maker/model details, warnings, tools, parts, or values.

Rules:
1. Answer the user's question directly.
2. Cite every factual claim using the exact format [p. X].
3. Preserve safety warnings and prerequisites from the manual.
4. If the excerpts are insufficient, say exactly what is missing.
5. Never present an inference as a confirmed manual instruction.
6. For safety-critical work, remind the user to verify the cited original page
   and follow vessel/company lockout-tagout and permit procedures.
"""

    user_prompt = f"""QUESTION:
{question}

MANUAL EXCERPTS:
{context}

Return:
- A concise answer
- A numbered procedure only if the manual excerpts clearly support one
- Relevant warnings
- The cited pages
"""

    message = client.messages.create(
        model=model_name,
        max_tokens=1200,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )


uploaded = st.file_uploader("Upload one PDF manual", type=["pdf"])

if uploaded:
    file_bytes = uploaded.getvalue()
    with st.spinner("Reading the manual..."):
        pages = extract_pdf(file_bytes)
        chunks = chunk_pages(pages)

    if not pages:
        st.error(
            "No selectable text was found. This PDF may be scanned; OCR is not included "
            "in this first prototype."
        )
        st.stop()

    st.success(
        f"Loaded **{uploaded.name}** — {len(pages)} text-bearing pages, "
        f"{len(chunks)} searchable chunks."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask about a maintenance procedure, maker/model, specification, warning, or part..."
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        evidence = retrieve(question, chunks, top_k)

        with st.chat_message("assistant"):
            if api_key:
                try:
                    with st.spinner("Searching and asking Claude..."):
                        answer = ask_claude(question, evidence, api_key, model)
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"Claude API error: `{exc}`"
                    st.error(answer)
            else:
                answer = (
                    "**Retrieval-only mode:** add an Anthropic API key in the sidebar "
                    "to generate a synthesized answer. The most relevant excerpts are below."
                )
                st.warning(answer)

            with st.expander("Show retrieved manual excerpts"):
                for i, item in enumerate(evidence, start=1):
                    st.markdown(
                        f"**Source {i} — page {item['page']} "
                        f"(retrieval score {item['score']:.3f})**"
                    )
                    st.write(item["text"])
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.markdown(
        """
### What this does
1. Extracts text from a PDF manual.
2. Searches for the passages most related to your question.
3. Sends only those passages to Claude.
4. Requires Claude to cite the original PDF pages.

### Good first test questions
- "What is the maintenance procedure for the fuel filter?"
- "What warnings apply before disassembly?"
- "Who is the maker and what model is listed?"
- "What tools or replacement parts are required?"
"""
    )
