import streamlit as st

from ingestion import (
    process_uploaded_pdf,
    get_indexed_pdfs,
    delete_pdf
)

from rag import (
    ask_question,
    get_rag_chain
)

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Multi PDF Research Assistant",
    page_icon="📚",
    layout="wide"
)

# =====================================================
# Session State
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("📚 Research Assistant")

    st.markdown("---")

    # -----------------------------------------
    # Upload PDFs
    # -----------------------------------------

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:

        st.write(
            f"{len(uploaded_files)} file(s) selected"
        )

        if st.button(
            "📥 Process Documents",
            use_container_width=True
        ):

            with st.spinner(
                "Processing documents..."
            ):

                for uploaded_file in uploaded_files:

                    result = process_uploaded_pdf(
                        uploaded_file
                    )

                    if result["status"] == "duplicate":

                        st.toast(
                            result["message"],
                            icon="📄"
                        )

                    else:

                        st.toast(
                            result["message"],
                            icon="✅"
                        )
            st.session_state.uploader_key += 1

            st.rerun()

    

    st.markdown("---")

    # -----------------------------------------
    # Indexed PDFs
    # -----------------------------------------

    st.subheader("Indexed PDFs")

    indexed_pdfs = get_indexed_pdfs()

    if len(indexed_pdfs) == 0:

        st.info(
            "No PDFs uploaded."
        )

    else:

        for pdf in indexed_pdfs:

            col1, col2 = st.columns(
                [4, 1]
            )

            with col1:

                st.caption(
                    f"📄 {pdf}"
                )

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{pdf}"
                ):

                    success = delete_pdf(pdf)

                    if success:

                        st.toast(
                            f"{pdf} deleted.",
                            icon="🗑️"
                        )
                    # Reset uploader
                    st.session_state.uploader_key += 1

                    st.rerun()

    st.markdown("---")

    # -----------------------------------------
    # Clear Chat
    # -----------------------------------------

    if st.button(
        "🧹 Clear Chat History",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.session_state.messages = []

        st.toast(
            "Chat history cleared.",
            icon="🧹"
        )

        st.rerun()

# =====================================================
# Main Page
# =====================================================

st.title(
    "📚 Multi PDF Research Assistant"
)

st.caption(
    "Upload PDFs and ask questions from them using RAG + Pinecone + Groq"
)


# =====================================================
# Display Chat Messages
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =====================================================
# User Input
# =====================================================

user_question = st.chat_input(
    "Ask a question about your documents..."
)

if user_question:

    # -----------------------------------------
    # User Message
    # -----------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )

    # -----------------------------------------
    # Assistant Response
    # -----------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Generating answer..."
        ):

            answer = ask_question(
                user_question,
                st.session_state.chat_history
            )

            st.markdown(
                answer
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # -----------------------------------------
    # Update Chat History
    # -----------------------------------------

    st.session_state.chat_history.append(
        (
            "human",
            user_question
        )
    )

    st.session_state.chat_history.append(
        (
            "ai",
            answer
        )
    )

