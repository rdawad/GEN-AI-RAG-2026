# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import os
# import json
# from langchain_chroma import Chroma
# import langchain
# from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer 
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from groq import Groq
# #from langchain.prompts import PromptTemplate
# from langchain_core.prompts import PromptTemplate
# #from langchain.chains import Retrieval
# #from langchain.chains import RetrievalQA
# from langchain_classic.chains import RetrievalQA
# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain

# print("all imports working fine")

# class RAGEngine:
#     '''langchain, chroma, groq , frontend- streamlit
#     chain-----backend system
#     1, load the document
#     2, create vector DB - chroma db
#     3, setup a chain(1,2)
#     4, call groq LLM
#     5  answer the query '''

#     # def __init__(self, json_path: str= "data.json", embeddings: str = "all-MiniLM-L6-v2"):
#     #     self.json_path = json_path
#     #     self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
#     #     self.vector_store = None
#     #     self.qa_chain = None

#     def __init__(self, json_path: str = "data.json"):
#         self.json_path = json_path
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )
#         self.vector_store = None
#         self.qa_chain = None

#     def load_document(self):
#         "load the json data from data.json file"
#         with open(self.json_path, 'r') as file:
#             data = json.load(file)

#         document = []

#         for item in data:
#             doc = Document(page_content = item['text'], metadata = {"id": item.get('id')})
#             document.append(doc)

#         return document
    
#     # def create_vectorstore(self, documents: list[Document]):
#     #     vector_store = Chroma(
#     #         embedding_function = self.embeddings,
#     #         collection_name="artifacts",
#     #         persist_directory="./chroma_db",
#     #     )
#     #     return self.vector_store
    
#     def create_vectorstore(self, documents: list[Document]):
#         self.vector_store = Chroma.from_documents(
#             documents=documents,
#             embedding=self.embeddings,
#             collection_name="artifacts",
#             persist_directory="./chroma_db",
#         )
#         return self.vector_store
    
#     from langchain_groq import ChatGroq

#     def setup_qa_chain(self):
#         '''
#         "setup a QA chain and call GROQ api
#         '''
#         groq_api_key = os.getenv("groq_api_key")
#         llm = ChatGroq(
#             groq_api_key=groq_api_key,
#             model="openai/gpt-oss-120b",
#             temperature=1
#         )

#         chat_completion = llm.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": "use the prompt to amswer the query"
                    
#                 }
#             ],
#             model="openai/gpt-oss-120b", # <-- Model specified here, not in Groq() init
#         )
#         prompt_template = """Use the following context to answer the question, if you don't know the answer, just say I don't have any answer for this question as of now, don't make up on answer
#         Context: {context}
#         Question: {question}
#         Answer:"""

#         PROMPT = PromptTemplate(
#             input_variables = ["context", "question"],
#             template = prompt_template
#         )
#         retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs = {"k": 3})
#         chain_type_kwargs = {"prompt": PROMPT}

#         self.qa_chain = RetrievalQA.from_chain_type(
#             chat_completion,
#             chain_type="stuff",
#             retriever=retriever,
#             chain_type_kwargs=chain_type_kwargs,
#             return_source_documents=True
#         )

#         return self.qa_chain
    
#     #from langchain_groq import ChatGroq

#     def setup_qa_chain(self):
#         llm = Groq(
#             groq_api_key=os.getenv("groq_api_key"),
#             model="openai/gpt-oss-120b",
#             temperature=1
#         )

#         PROMPT = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
#             Use the following context to answer the question.
#             If you don't know the answer, say you don't know.

#             Context: {context}
#             Question: {question}
#             Answer:
#             """
#         )

#         retriever = self.vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 3}
#         )

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever,
#             chain_type_kwargs={"prompt": PROMPT},
#             return_source_documents=True
#         )

#         return self.qa_chain
    
#     def initialize(self):
#         "initialize the RAG engine"
#         document = self.load_document()
#         self.create_vectorstore(document)
#         self.setup_qa_chain()
#         print("Rag engine initialized successfully.")

#     def answer_query(self, question:str) -> dict:
#         """Answer a quesry using QA chain"""
#         if not self.qa_chain:
#             raise ValueError("QA chain is not initialized call initialize() first")

#         result = self.qa_chain.invoke({"query": question})

#         retn =  {
#             "answer": result["result"],
#             "source_documents": [
#                 {
#                     "content": doc.page_content,
#                     "metadata": doc.metadata
#                 }
#                 for doc in result["source_documents"]
#             ]
#         }
#         return retn




    

        
    


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


class RAGEngine:

    def __init__(self, json_path: str = "data.json"):
        self.json_path = json_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None

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

    def setup_qa_chain(self):
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b",
            temperature=1,
            max_completion_tokens=8192,
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question.
            If you don't know the answer, just use your own knowledge base from LLM or search it from google and get the best answer.

            Context: {context}
            Question: {question}
            Answer:
            """
        )

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return self.qa_chain

    def initialize(self):
        documents = self.load_document()
        self.create_vectorstore(documents)
        self.setup_qa_chain()
        print("RAG engine initialized successfully")

    def answer_query(self, question: str):
        if not self.qa_chain:
            raise ValueError("Call initialize() first")

        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }




    