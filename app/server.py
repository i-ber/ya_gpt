from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

import os

from yandex_chain import YandexEmbeddings, ChatYandexGPT, YandexLLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.pydantic_v1 import BaseModel

app = FastAPI(
    title="YaGPT LangServe Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


ya_auth = {
    "api_key": os.getenv("YA_API_KEY"),
    "folder_id": os.getenv("YA_FOLDER_ID")
}

collection_name = "botpress_collection"

namespace = "test1"
system_prompt = """Ты ассистент отеля, в твоей базе есть следующая информация по вопросу клиента,
ты можешь её использовать для ответа:
{context}
"""


embeddings = YandexEmbeddings(**ya_auth)
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMADB_HOST"),
    port=int(os.getenv("CHROMADB_PORT")),
    settings=Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                      chroma_client_auth_credentials=os.getenv("CHROMA_AUTH_TOKEN")))

langchain_chroma = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)
retriever = langchain_chroma.as_retriever(search_kwargs={'filter': {'hotel': 'bridgeresort'}}).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="Metadata filter for searching docs"
    )
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


default_template = """Ты ассистент отеля, в твоей базе есть следующая информация по вопросу клиента,
ты можешь её использовать для ответа:
{context}

Вопрос: {input}

Ответ:"""
prompt = PromptTemplate.from_template(default_template).configurable_fields(
    template=ConfigurableField(
        id="prompt",
        name="Prompt",
        description="The prompt template to use context and input",
    )
)
llm = YandexLLM(**ya_auth, temperature=0.5, max_tokens=100, use_lite=False).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    ),
    max_tokens=ConfigurableField(
        id="llm_max_tokens",
        name="LLM Max tokens",
        description="The max nubmer of tokens in the LLM response",
    ),
    use_lite=ConfigurableField(
        id="llm_use_lite",
        name="LLM lite model",
        description="Lite model flag",
    )
)
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
).with_config(
    tags=["contextualize_q_chain"]
)


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
