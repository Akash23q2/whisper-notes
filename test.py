# from langchain_google_genai import ChatGoogleGenerativeAI
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Load your API key securely from environment
# os.environ["GOOGLE_API_KEY"] = "AIzaSyAmF_50nipjgtNuRuBoeH5b1yY-nTPr6Rc" 
# print("hello world")
# llm=ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
# )
# result=llm.invoke("hey there! hello world !")
# print(result.content)
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from os import getenv  
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

model = GoogleModel(
    # 'cognitivecomputations/dolphin-mistral-24b-venice-edition:free',
    'gemini-2.5-flash',
    # provider=OpenRouterProvider(api_key=getenv("OPENROUTER_API_KEY")),
)
agent = Agent(model)
def add_context(ai:str="",query:str="hey there! hello world !") -> str:
    return f"context: ai: {ai}, query: {query}\n "
context=""
async def main() :
    while True:
        # print("\033[2J\033[H")
        async with agent.run_stream(input("Enter Query: ")) as stream:
            ans=await stream.get_output()
            print(ans)

if __name__ == "__main__":
    asyncio.run(main())

    

