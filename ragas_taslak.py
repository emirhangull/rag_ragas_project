import os
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
import warnings
warnings.filterwarnings("ignore")

openai_client = OpenAI(
    api_key="EMPTY",
    base_url=settings.vllm_base_url,
)

ragas_llm = llm_factory(settings.vllm_model, client=openai_client)

langchain_embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model_name
)
ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm = ragas_llm
context_recall.llm = ragas_llm

data = {
    "question": ["Ragas nedir?"],
    "answer": ["Ragas, RAG sistemlerini değerlendirmek için bir çerçevedir."],
    "contexts": [["Ragas, RAG uygulamaları için değerlendirme sağlayan bir kütüphanedir."]],
    "ground_truth": ["Ragas, RAG sistemlerini değerlendirmek için kullanılan bir kütüphanedir."]
}
dataset = Dataset.from_dict(data)

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

result = evaluate(
    dataset=dataset,
    metrics=metrics,
)

df = result.to_pandas()
print(df)

print(df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].to_string())