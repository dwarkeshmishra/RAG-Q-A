import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("API key is not set. Please add GROQ_API_KEY to your .env file.")
    st.stop()

# Define paths
KNOWLEDGE_BASE_DIR = os.path.join("knowledge_base")
VECTORSTORE_DIR = os.path.join("vectorstore")

# Ensure directories exist
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

@st.cache_resource
def setup_vectorstore():
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        if os.path.exists(KNOWLEDGE_BASE_DIR) and os.listdir(KNOWLEDGE_BASE_DIR):
            loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="*.txt", loader_cls=TextLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
            docs = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(VECTORSTORE_DIR)
        else:
            st.warning(f"No knowledge base files found in {KNOWLEDGE_BASE_DIR}. Q&A will be limited to LLM's pre-trained knowledge.")
            return None
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = setup_vectorstore()

qa_chain = None
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.2)
if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True
    )
else:
    # Fallback to LLM without RAG if no vectorstore is available
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    template = """You are a financial assistant specializing in loan approvals. Answer the following question based on general financial knowledge:
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    qa_chain = LLMChain(llm=llm, prompt=prompt)

st.set_page_config(page_title='Loan Approval RAG Bot', page_icon='💬')
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []

st.title('💬 Loan Approval Assistant')
st.markdown("Ask questions about loan approvals or input applicant details for guidance on eligibility based on general financial knowledge.")

# Display chat history
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat['query'])
    with st.chat_message("assistant"):
        st.write(chat['answer'])

# Input for user query
user_query = st.chat_input('Type your question or input applicant details...')
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if vectorstore:
                result = qa_chain({'query': user_query})
                st.write(result['result'])
            else:
                result = qa_chain({'question': user_query})
                st.write(result['text'])
    st.session_state.history.append({'query': user_query, 'answer': result['result'] if vectorstore else result['text']})
