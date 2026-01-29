# A streamlit frontend for RAG with Groq API and OpenAI
import streamlit as st
import os
import json
from backend_system.ragengine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG with Groq API and OpenAI", 
    layout="wide",
)

@st.cache_resource
def initialize_rag_engine():
    rag_engine = RAGEngine("data.json")
    rag_engine.initialize()
    return rag_engine

## main app
def main():
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False

    st.sidebar.title("Rag with GROQ API and OPENAI")

    if st.sidebar.button("Initialize RAG Engine"):
        with st.spinner("Initializing RAG Engine...."):
            rag_engine = initialize_rag_engine()
            st.session_state.rag_initialized = True
        st.sidebar.success("RAG Engine initialized successfully!")
    
    if st.session_state.rag_initialized:
        rag_engine = initialize_rag_engine()
        question = st.text_input("Enter your question")
        if st.button("Get Answer"):
            if question.strip == "":
                st.warning("Please enter a valid question")
            else:
                with st.spinner("Getting answer..."):
                    response = rag_engine.answer_query(question)
                    st.subheader("Answer:")
                    st.write(response["answer"])

    # rag_initialized = True
    # st.button
    # rag = initialize_rag_engine()




if __name__ == "__main__":
    main()