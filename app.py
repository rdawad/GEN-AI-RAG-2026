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
    # Initialize session state
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = "default"

    st.sidebar.title("RAG with GROQ API and OPENAI")

    # Initialize RAG Engine
    if st.sidebar.button("Initialize RAG Engine"):
        with st.spinner("Initializing RAG Engine...."):
            rag_engine = initialize_rag_engine()
            st.session_state.rag_initialized = True
        st.sidebar.success("RAG Engine initialized successfully!")
    
    # Clear Memory Button
    if st.sidebar.button("Clear Chat History"):
        if st.session_state.rag_initialized:
            rag_engine = initialize_rag_engine()
            rag_engine.clear_memory(st.session_state.session_id)
            st.session_state.chat_history = []
            st.sidebar.success("Chat history cleared!")
        else:
            st.sidebar.warning("Please initialize RAG Engine first!")
    
    # Main chat interface
    if st.session_state.rag_initialized:
        rag_engine = initialize_rag_engine()
        
        st.title("💬 Chat with Your Documents")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
        
        # Question input
        question = st.text_input("Enter your question:", key="question_input")
        
        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a valid question")
            else:
                with st.spinner("Getting answer..."):
                    response = rag_engine.answer_query(
                        question, 
                        session_id=st.session_state.session_id
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response["answer"]
                    })
                    
                    # Display the new answer
                    with st.chat_message("user"):
                        st.write(question)
                    with st.chat_message("assistant"):
                        st.write(response["answer"])
                    
                    # Show source documents in expander
                    if response["source_documents"]:
                        with st.expander("📄 View Source Documents"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Document {i+1}:**")
                                st.write(doc["content"])
                                st.markdown(f"*Metadata: {json.dumps(doc['metadata'])}*")
                                st.divider()
                    
                    # Rerun to update chat history display
                    st.rerun()
    else:
        st.info("👈 Please initialize the RAG Engine from the sidebar to start chatting!")

if __name__ == "__main__":
    main()