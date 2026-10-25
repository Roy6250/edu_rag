"""RAGAS evaluation: faithfulness, answer relevancy, context recall, context precision."""
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

import config
from rag import EduRAG


def run_eval(csv_path: str, llm_model: str = config.LLM_MODEL, out_csv: str | None = None) -> dict:
    """csv must have columns: question, ground_truth"""
    df = pd.read_csv(csv_path)
    rag = EduRAG(llm_model=llm_model)

    rows = []
    for i, r in df.iterrows():
        out = rag.answer(r["question"])
        rows.append({
            "user_input": r["question"],
            "response": out["answer"],
            "retrieved_contexts": out["contexts"],
            "reference": r["ground_truth"],
        })
        print(f"[{i + 1}/{len(df)}] answered")

    judge = LangchainLLMWrapper(ChatOpenAI(model=config.EVAL_LLM_MODEL, api_key=config.OPENAI_API_KEY))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=config.OPENAI_API_KEY))
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall(), LLMContextPrecisionWithReference()],
        llm=judge,
        embeddings=emb,
    )

    scores = result.to_pandas()
    if out_csv:
        scores.to_csv(out_csv, index=False)
    summary = scores.select_dtypes("number").mean().round(4).to_dict()

    print(f"\nMetric scores for {llm_model}")
    for name, val in summary.items():
        print(f"  {name:<40} {val}")
    return summary
