# https://python.langchain.com/docs/modules/agents/tools/custom_tools
import os

from langchain.agents import AgentExecutor, create_structured_chat_agent, create_openai_functions_agent, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from typing import Optional, Type
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from glob import glob
from pathlib import Path
from langchain.agents import Tool
from langchain.tools import tool   
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


llm = ChatOpenAI(temperature=0,
        model_name='gpt-3.5-turbo',
        openai_api_key=os.environ["OPENAI_API_KEY"],)


embeddings = OpenAIEmbeddings(
        model = "text-embedding-ada-002", 
)

vectorstore = FAISS.load_local(folder_path="vecstore/state_of_the_union_2023/", embeddings=embeddings, allow_dangerous_deserialization=True)

@tool
def stateoftheunion_tool(question:str):
    """Useful to get information about the state of the union"""
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever()),


from tools.state_of_the_union_vestore import get_tools
tools = get_tools(embeddings=embeddings)

from tools.github_releases import get_toolkit

from tools import github_releases
tools = tools + github_releases.get_toolkit()
#tools = [stateoftheunion_tool]
#tools = []
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful agent.
        You can use your tools to answer questions. If you do not have a tool to
        answer the question, say so. 
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(prompt=prompt, tools=tools, llm=llm)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = conversational_memory)

#out = agent_executor.invoke({"input": "What is most important in the state of the union?", "chat_history": conversational_memory.chat_memory.messages})
out = agent_executor.invoke({"input": "What is the latest version of repo satpy with owner pytroll on github and what can you tell me about it, specially what bugs were fixed?", "chat_history": conversational_memory.chat_memory.messages})

print(out['output'])