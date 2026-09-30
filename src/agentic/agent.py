import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from .service.tools import search
from langchain.messages import HumanMessage,AIMessage

class Agent:
    def __init__(self) -> None:
        model=init_chat_model(
            model="deepseek-chat",
            model_provider="openai",
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        self.model=model
        self.agent=create_agent(
            model=model,
            tools=[search],
            system_prompt=
            '''
            你是一个专业的教师资格证考试辅导专家。您的主要职责是帮助用户解决有关教资考试的相关问题。产品说明:
            1。如果用户问了一个你不确定的问题，或者涉及教资专业知识的问题，你必须使用`search_knowledge_base`工具来查阅相关文档。
            2. 在引用文档时，要清楚地总结包括内容中的相关上下文。
            3. 如果获取文档失败，请告诉用户，并以您最好的专家理解继续进行。
            在回答用户关于教资考试知识的问题之前，您必须查阅工具以获取最新信息。你的回答应该清晰、简洁、准确。不要有过多解释除非用户询问。
            ''',
        )

    def invoke(self,query:str):
        response=self.agent.stream(
            {
                "messages":[HumanMessage(content=query)]
            },
            stream_mode="messages"
        )
        for chunk,metadata in response:
            print(chunk.content,flush=True)
