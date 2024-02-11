from langchain.agents import AgentExecutor, create_react_agent, Tool, create_structured_chat_agent
from langchain_openai import ChatOpenAI
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = TextLoader(
    "./state_of_the_union_2023.txt"
).load()

docs = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
).split_documents(text)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(), persist_directory="./chroma_db"
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)
def run_qa_chain(question):
    print("Inside vector fchain")
    results = qa_chain.invoke({"question":question},return_only_outputs=True)
    return str(results)


tool = Tool(
            name="State of the Union 2023 QA System",
            func=run_qa_chain,
            description="Useful for when you need to answer questions about "
                        "the most recent state of the union address. Input "
                        "should be a fully formed question.",
            return_direct=True,
        )


from langchain.pydantic_v1 import BaseModel, Field
from typing import Union, Type
from langchain.tools import BaseTool

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from typing import Optional, Type

class VectorStoreInput(BaseModel):
    query: str = Field(description="should be a search query")

class VectorStoreTool(BaseTool):
    name = "VectorStore"
    description = "useful to look into the vectorstore"
    args_schema: Type[BaseModel] = VectorStoreInput
    return_direct: bool = True
    def _run_chain(question):
        print("Inside vector fchain")
        results = qa_chain.invoke({"question":question},return_only_outputs=True)
        return str(results)
    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        print("Doing vector store fetch")
        """Use the tool."""
        
        return run_qa_chain(question=query)
    
    async def _arun(
            self,
            a: int,
            b: int,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("Vector store does not support async")

tools = [VectorStoreTool()]

llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        model_name='gpt-3.5-turbo'
)

from langchain import hub
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

#prompt = hub.pull("hwchase17/react")
prompt = hub.pull("hwchase17/structured-chat-agent")

conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)
from langchain.agents import initialize_agent, AgentType
#agent = initialize_agent(prompt=prompt, tools=tools, llm= llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
agent = create_structured_chat_agent(prompt=prompt, tools=tools, llm=llm)
#agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = conversational_memory)
#agent_executor.invoke("can you calculate the circumference of a circle that has a radius of 7.81mm")
#out = agent_executor.invoke({"input": "can you calculate the circumference of a circle that has a radius of 7.81 mm and 10.2 mm", "chat_history": []})
#out = agent_executor.invoke({"input":"Tell me about the state of the union"})
#out = agent_executor.invoke({"input":"How much is 10 bitcoins in GBP?"})
#out = agent_executor.invoke({"input": "List all titles mentioned in the State of the union", "chat_history": []})
out = agent_executor.invoke({"input":"Summarize the state of the union"})
print(out)

exit(0)
out = agent_executor.invoke({"input": "What do you get when you multply 2 and 4?", "chat_history": []})
print(out)
exit(0)
from langchain.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print("Doing mathy things")
    return a * b

calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
)
m = calculator
print(m.name)
print(m.description)
print(m.args)

print(calculator.run({"a":1, "b":3}))

#os.open('prompttest.json')
