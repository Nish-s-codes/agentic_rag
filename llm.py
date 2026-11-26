import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ["LLM_API_KEY"],
    model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
    temperature=0.2,
)

if __name__ == "__main__":
    print(llm.invoke("Say hello in one short sentence.").content)