import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "my-rag-index"

# Инициализация клиента Pinecone
client = pinecone.Client(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Создаем индекс, если нет
if INDEX_NAME not in client.list_indexes():
    client.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )

# Получаем индекс
index = client.Index(INDEX_NAME)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

vectorstore = LangchainPinecone(index, embedding.embed_query, text_key="text")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

query = "Когда была основана компания Apple?"
print(qa.run(query))




