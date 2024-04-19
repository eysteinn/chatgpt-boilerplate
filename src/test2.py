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


def load_key():
    return os.environ.get("PROMPT_APP_KEY") or os.environ.get("PROMPT_KEY")

def detect_user():
    return os.environ.get("OPENAI_USER")
    
llm = ChatOpenAI(model_name= "gpt-4-32k-0314", # "gpt-4", #"gpt-4-turbo-preview", #",
                 openai_api_base = "https://llm-proxy-api.ai.openeng.netapp.com",
                 openai_api_key  = load_key(),
                 http_client=httpx.Client(verify=False),
                 model_kwargs = {'user': detect_user()})

embeddings = OpenAIEmbeddings(
        model           = "text-embedding-ada-002", # "text-embedding-ada-002",
        http_client     = httpx.Client(verify=False),
        openai_api_base = "https://llm-proxy-api.ai.openeng.netapp.com",
        openai_api_key  = load_key(),
        model_kwargs    = {'user': detect_user() }) 


vectorstore = FAISS.load_local(folder_path="/Users/eysteinn/projects/python/HAL9000/trainingset/faiss_combined", embeddings=embeddings, allow_dangerous_deserialization=True)

folder_path = "userspace/*"
print('Looking at ', folder_path)
for tmp in glob(folder_path):
    if Path(tmp).is_dir():
        #print('Adding: {}'.format(tmp))
        tmpdb = FAISS.load_local(folder_path=tmp, embeddings=embeddings, allow_dangerous_deserialization=True)
        vectorstore.merge_from(tmpdb)

#vectorstore = FAISS.load_local(folder_path="/Users/eysteinn/projects/python/HAL9000/userspace/confluence_248356920", embeddings=embeddings, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)
print("Chain created")
class DocumentInput(BaseModel):
    question: str = Field()
    
t1 = Tool(
    args_schema=DocumentInput,
    name='DocumentQA',
    description=f"useful for when you need to answer questions about anything related to nkdev",
    func=RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever()),
    )



class ReleaseTagInput(BaseModel):
    tag: str = Field(description="release version in semver format (e.g. 1.2.3), only the version number should be passed")
from test_github import Fetch_release, Fetch_latest


from test_github import Fetch_latest, Fetch_release
githubversion_tool = Tool(
    #args_schema=ReleaseTagInput,
    args_schema = ReleaseTagInput,
    name="GithubVersion",
    description="useful when you need to answer question about particular nkdev release or versions",
    func=Fetch_release,
    #return_direct=True,
)

class ReleaseLatestTagInput(BaseModel):
    pass
    
githublatest_tool = Tool(
    #args_schema=ReleaseTagInput,
    #args_schema = ReleaseTagInput,
    args_schema = ReleaseLatestTagInput,
    name="GithubLatestVersion",
    description="useful when you need to answer question about the latest nkdev release or versions",
    func=Fetch_latest,
    #return_direct=True,
)


from langchain.tools.base import StructuredTool
githublatest_tool = StructuredTool.from_function(func=Fetch_latest, name="GithubLatestVersion", description="useful when you need to answer question about the latest nkdev release or versions")
    

from langchain.tools import tool   

from tools.github_releases import github_version
#@tool
#def github_version(tag: str):
#    """Gets a certain version or release of nkdev"""
#    return Fetch_release(tag)

@tool
def githublatest_tool(repo:str = "nkdev"):
    """Gets the latest release of a repository, default is nkdev"""
    return Fetch_latest(repo)

#class DocumentInput(BaseModel):
#    question: str = Field(description="release version in semver format (e.g. 1.2.3), only the version number should be passed")


class VectorStoreInput(BaseModel):
    a: str = Field(description="the question")
    
class VectorStoreTool(BaseTool):
    name = "VectorStore"
    description = "useful for when you need to answer questions about anything related to nkdev"
    args_schema: Type[BaseModel] = VectorStoreInput
    return_direct: bool = True
    def _run(
            self, a: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        print("Checking vectorstore")
        """Use the tool."""
        results = qa_chain.invoke({"query":a},return_only_outputs=True)

        import json
        #return json.dumps(results)
        print(results)
        return json.dumps(results['result'])
    
    async def _arun(
            self,
            a: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("Calculator does not support async")



from langchain.agents import AgentType, initialize_agent
#from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
#from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import OpenAI
#toolkit = JiraToolkit.from_jira_api_wrapper(jira)

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    #return_direct: bool = True
    def _run(
            self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        print("Doing multiplication")
        """Use the tool."""
        import json
        return json.dumps(a*b)
    
    async def _arun(
            self,
            a: int,
            b: int,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("Calculator does not support async")


tools = [CustomCalculatorTool(), t1] # VectorStoreTool()]
#tools.append(toolkit.get_tools())
#tools = toolkit.get_tools()
#tools=tools+[githublatest_tool, githubversion_tool]
#tools=tools+[github_version]
from tools.github_releases import get_toolkit
tools = tools + get_toolkit()

from langchain import hub
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

#prompt = hub.pull("hwchase17/structured-chat-agent")
prompt = hub.pull("hwchase17/openai-tools-agent")
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a tech support agent, mainly assisting with Azure and the software nkdev.
        You can use your tools to answer questions. If you do not have a tool to
        answer the question, say so. 
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)
agent = create_openai_tools_agent(prompt=prompt, tools=tools, llm=llm)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = conversational_memory)

#out = agent_executor.invoke({"input": "What do you get when you multply 2 and 4?", "chat_history": []})
#print(out)
#out = agent_executor.invoke({"input": "is there a way to copy the file into nkdev node in azure?", "chat_history": conversational_memory.chat_memory.messages})
#out = agent_executor.invoke({"input": "How do I install cvs-sde using the latest version of nkdev?", "chat_history": conversational_memory.chat_memory.messages})
out = agent_executor.invoke({"input": "How do I install cvs-sde?", "chat_history": conversational_memory.chat_memory.messages})

print(conversational_memory.chat_memory.messages)
#out = agent_executor.invoke({"input": "Tell me about nkdev version 1.8.1 and 1.6.9 and anf-resource-provider?", "chat_history": conversational_memory.chat_memory.messages})
#out = agent_executor.invoke({"input": "list the first 3 releases of nkdev and tell me a little bit about each one?", "chat_history": conversational_memory.chat_memory.messages})
#out = agent_executor.invoke({"input": "Who contributed to nkdev release 1.8.8?", "chat_history": conversational_memory.chat_memory.messages})
#out = agent_executor.invoke({"input": "What is the speed of light?", "chat_history": conversational_memory.chat_memory.messages})


#out = agent_executor.invoke({"input": "Tell me about the latest versions of nkdev and anf-resource-provider?", "chat_history": conversational_memory.chat_memory.messages})
#out = agent_executor.invoke({"input": "What can you tell me about the jira ticket NFSAAS-82416?", "chat_history": conversational_memory.chat_memory.messages})
#out = agent_executor.invoke({"input": "What can you tell me about nkdev release v1.6.8?", "chat_history": conversational_memory.chat_memory.messages})
#print('output: ',out['output'])
#out = agent_executor.invoke({"input": "What can you tell me about the latest version of nkdev?", "chat_history": conversational_memory.chat_memory.messages})
#print('output: ',out['output'])
#out = agent_executor.invoke({"input": "What have we discussed so far?", "chat_history": conversational_memory.chat_memory.messages})
#print(out)
