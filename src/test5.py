from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_openai import ChatOpenAI
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

import os


vectorstore = Chroma(persist_directory="./chroma_db")

def run_qa_chain(question):
    print('Hallo')
    results = qa_chain({"question":question},return_only_outputs=True)
    return str(results)


tool = Tool(
            name="State of the Union 2023 QA System",
            func=run_qa_chain,
            description="Useful for when you need to answer questions about "
                        "the most recent state of the union address. Input "
                        "should be a fully formed question.",
            return_direct=True,
        )
tools = [tool]

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
out = agent_executor.invoke({"input": "can you calculate the circumference of a circle that has a radius of 7.81mm", "chat_history": []})
