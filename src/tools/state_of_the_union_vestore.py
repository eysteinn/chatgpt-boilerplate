#from langchain.tools import tool   
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from typing import List
from langchain_community.tools import BaseTool
#from langchain.chains import RetrievalQA

from langchain.tools.retriever import create_retriever_tool




#@tool
#def stateoftheunion_tool(question:str):
#    """Useful to get information about the state of the union"""
#    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever()),

#def get_tools(embeddings: OpenAIEmbeddings) -> List[BaseTool]:
#    vectorstore = FAISS.load_local(folder_path="vecstore/state_of_the_union_2023/", embeddings=embeddings, allow_dangerous_deserialization=True)


from langchain.tools.retriever import create_retriever_tool

def get_tools(embeddings: OpenAIEmbeddings) -> List[BaseTool]:
    vectorstore = FAISS.load_local(folder_path="vecstore/state_of_the_union_2023/", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "search_state_of_union",
        "Searches and returns excerpts from the 2022 State of the Union.",
    )
    return [tool]