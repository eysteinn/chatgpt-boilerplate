from langchain.agents import AgentExecutor, create_react_agent, Tool, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#https://python.langchain.com/docs/integrations/vectorstores/faiss
#https://python.langchain.com/docs/integrations/toolkits/document_comparison_toolkit

def file2vecstore(filename):
    text = TextLoader(filename).load()
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    ).split_documents(text)
    db = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    return db

def save_vecstore(db, filename):
    from pathlib import Path
    fp = Path(filename)
    dbfile = Path('vecstore') / fp.stem
    db.save_local(dbfile)
    return dbfile.absolute()

def db2retreivalchain(db):
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    qa_chain.invoke({"question":"summarize the state of the union"},return_only_outputs=True)
    return qa_chain

    #answer = retriever.invoke("What was the state of the union about?")



filename= "./state_of_the_union_2023.txt"
file2vecstore(filename)

