from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load your API key securely from environment
os.environ["GOOGLE_API_KEY"] = "AIzaSyAmF_50nipjgtNuRuBoeH5b1yY-nTPr6Rc" 
print("hello world")
llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)
result=llm.invoke("hey there! hello world !")
print(result.content)
