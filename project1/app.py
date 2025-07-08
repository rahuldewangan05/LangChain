### for basic ui
import streamlit as st 
import os

## for using open source api key
from langchain_groq import ChatGroq

## embeddings and text splittter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## for chain and prompts
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

## vectorscore
from langchain_community.vectorstores import FAISS

## loading local local env
from dotenv import load_dotenv


load_dotenv()

## load groq api key
groq_api_key=os.getenv("GROQ_API_KEY")

### title of project
st.title("Chatgroq with Llama3 Demo")


## loading llama 3 model using groq
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    )


## prompt to send to llm
prompt=ChatPromptTemplate.from_template(
    """ 
    Answer the question based on the provided context only.
    Pleaase provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    
    """
)

prompt1=st.text_input("Enter your question from Documents")


## loading hugging face embeddding
embedding=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

## function to create embedding and storing in vectorstore
def vector_embedding():
    if "vectordb" not in st.session_state:
        st.session_state.loader=PyPDFDirectoryLoader("./us-census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.document=st.session_state.splitter.split_documents(documents=st.session_state.docs)
        st.session_state.vectordb=FAISS.from_documents(st.session_state.document,embedding)     
    

## creating a button to perfomr task such as reading provided documents
## creating embeddings, and store it in vector store to answer the question asked
if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vectorstore DB is ready")
    
    
## for prompt1
import time
if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectordb.as_retriever()
    retriver_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retriver_chain.invoke({'input':prompt1})
    print("Response time ",time.process_time()-start)
    st.write(response['answer'])
    
    ## also writing the potion of document used to generating response
    with st.expander("Document Similarity Search"):
        ## find the relevant chunks
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------------")