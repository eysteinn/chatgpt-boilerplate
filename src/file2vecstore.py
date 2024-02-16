from langchain.agents import AgentExecutor, create_react_agent, Tool, create_structured_chat_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#https://python.langchain.com/docs/integrations/vectorstores/faiss
#https://python.langchain.com/docs/integrations/toolkits/document_comparison_toolkit

def getEmbeddings():
    api_type = os.getenv("OPENAI_API_TYPE")
    
    if api_type == "azure":
        print("Embeddings type: 'azure'")
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(azure_endpoint="https://openlab-openai.openai.azure.com/")
    elif api_type == "llama2":
        print("Embeddings type: 'llama2'")
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings()

    else:
        print("Embeddings type: 'openai'")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

def getLLM():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise Exception("OPENAI_API_KEY not set")
    api_version = os.getenv("OPENAI_API_VERSION")
    
    api_type = os.getenv("OPENAI_API_TYPE")
    if api_type == "azure":
        print("LLM type: 'azure'")
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", 
                               api_key=api_key, api_version=api_version, 
                               azure_endpoint="https://openlab-openai.openai.azure.com/",
                               azure_deployment="gpt-4-32k-0314") 
    elif api_type == "llama2":
        print("LLM type: 'llama2'")
        from langchain_community.llms import Ollama
        model = "llama2"
        llm = Ollama(model=model)
        return llm
    else:
        print("LLM type: 'openai'")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)
    
    
def file2vecstore(filename):
    text = TextLoader(filename).load()
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    ).split_documents(text)
    db = FAISS.from_documents(docs, embedding=getEmbeddings())
    return db

def save_vecstore(db, filename):
    from pathlib import Path
    fp = Path(filename)
    dbfile = Path('vecstore') / fp.stem
    db.save_local(dbfile)
    return dbfile.absolute()

def load_vecstore(filename):
    #api_ver=None
    #if os.environ.get("OPENAI_API_TYPE") == "azure":
    #    api_ver = os.environ.get("OPENAI_API_VERSION")
    db = FAISS.load_local(filename, embeddings=getEmbeddings())
    return db
     
def db2retreivalchain(db):
    if os.environ.get("OPENAI_API_TYPE") == "llama2":
    #     print('Using llama2')
        from langchain.chains import RetrievalQA
        qa_chain =  RetrievalQA.from_chain_type(llm=getLLM(), chain_type = "stuff",return_source_documents=True, retriever=db.as_retriever())
    else:
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=getLLM(),
            chain_type="stuff",
            retriever=db.as_retriever(),
        )
    #qa_chain.invoke({"question":"summarize the state of the union"},return_only_outputs=True)
    return qa_chain

    #answer = retriever.invoke("What was the state of the union about?")



if __name__ == "__main__":
    import sys
    for file in sys.argv[1:]:
        db = file2vecstore(file)
        dbfilename = save_vecstore(db, file)
        print("Saved vecstore to ", dbfilename)
        chain = db2retreivalchain(db)
        ret = chain.invoke("Summarize the state of the union address")
        print(ret["answer"])

    if len(sys.argv) == 1:
        filename='vecstore/state_of_the_union_2023'
        db = load_vecstore(filename)
        chain = db2retreivalchain(db)
        ret = chain.invoke("Summarize the state of the union address")