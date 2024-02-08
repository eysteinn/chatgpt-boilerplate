
# https://python.langchain.com/docs/modules/agents/tools/custom_tools
import os

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from typing import Optional, Type

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True
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


tools = [CustomCalculatorTool()]

llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        model_name='gpt-3.5-turbo'
)

from langchain import hub
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

prompt = hub.pull("hwchase17/structured-chat-agent")

conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)
agent = create_structured_chat_agent(prompt=prompt, tools=tools, llm=llm)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = conversational_memory)
out = agent_executor.invoke({"input": "What do you get when you multply 2 and 4?", "chat_history": []})

print(out)