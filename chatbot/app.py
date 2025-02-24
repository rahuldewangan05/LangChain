from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()



api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Please set it before running the script.")
os.environ["OPENAI_API_KEY"] = api_key

# LangSmith tracking
langsmith_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"


os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
#LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")


#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("human","Question:{question}")
    ]
)

#streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text=st.text_input("Search the topic you want")

# OpenAI LLM
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))