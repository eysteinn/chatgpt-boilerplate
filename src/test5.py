from langchain.agents import AgentExecutor, create_react_agent, Tool
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
    print("Inside vector chain")
    results = qa_chain.invoke({"question":question},return_only_outputs=True)
    return str(results)


from langchain.tools import BaseTool
from math import pi
from typing import Union
class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle, pass only the radius and not text and no units"

    def _run(self, radius: Union[int, float]):
        print("inside Circtool", radius)
        
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")
    
tool = Tool(
            name="State of the Union 2023 QA System",
            func=run_qa_chain,
            description="Useful for when you need to answer questions about "
                        "the most recent state of the union address. Input "
                        "should be a fully formed question.",
            return_direct=True,
        )
tools = [tool, CircumferenceTool()]

llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        model_name='gpt-3.5-turbo'
)

from langchain import hub
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

prompt = hub.pull("hwchase17/react")
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = conversational_memory)
#agent_executor.invoke("can you calculate the circumference of a circle that has a radius of 7.81mm")
#out = agent_executor.invoke({"input": "can you calculate the circumference of a circle that has a radius of 7.81 mm and 10 mm", "chat_history": []})
out = agent_executor.invoke({"input": "List all titles mentioned in the State of the union", "chat_history": []})