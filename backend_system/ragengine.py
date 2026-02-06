__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Updated imports for LangChain 0.1+
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGEngine:

    def __init__(self, json_path: str = "data.json"):
        self.json_path = json_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None
        self.chat_history_store = {}  # Store chat histories per session

    def load_document(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)

        return [
            Document(page_content=item["text"], metadata={"id": item.get("id")})
            for item in data
        ]

    def create_vectorstore(self, documents):
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="artifacts",
            persist_directory="./chroma_db",
        )
        return self.vector_store

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = ChatMessageHistory()
        return self.chat_history_store[session_id]

    def format_docs(self, docs):
        """Format retrieved documents into a string"""
        return "\n\n".join(doc.page_content for doc in docs)

    def setup_qa_chain(self):
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b",
            temperature=1,
            max_completion_tokens=8192,
        )

        # Create a prompt with chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer questions.
            If you don't know the answer from the context, use your own knowledge.
            
            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Create a chain that retrieves context based on input
        def get_context(x):
            return self.format_docs(retriever.invoke(x["input"]))
        
        # Create the RAG chain
        rag_chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Wrap with message history
        self.qa_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        # Store retriever for getting source documents
        self.retriever = retriever

        return self.qa_chain

    def initialize(self):
        documents = self.load_document()
        self.create_vectorstore(documents)
        self.setup_qa_chain()
        print("RAG engine initialized successfully")

    def answer_query(self, question: str, session_id: str = "default"):
        if not self.qa_chain:
            raise ValueError("Call initialize() first")

        # Get the answer - pass as dictionary with "input" key
        answer = self.qa_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Get source documents
        source_docs = self.retriever.invoke(question)

        return {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
        }
    
    def clear_memory(self, session_id: str = "default"):
        """Clear the conversation memory for a session"""
        if session_id in self.chat_history_store:
            self.chat_history_store[session_id].clear()
            print(f"Memory cleared for session: {session_id}")
    
    def get_chat_history(self, session_id: str = "default"):
        """Get the current chat history for a session"""
        if session_id in self.chat_history_store:
            return self.chat_history_store[session_id].messages
        return []
    

# 1 ChatMessageHistory
# 2. RunnableWithMessageHistory
# 3. three methods (get session history, clear-chat-history, get-chat-history)
# 4. changes a qa_chain setup ()