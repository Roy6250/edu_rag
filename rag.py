"""Question answering and question generation over the knowledge base."""
import json
import re

from llama_index.core import PromptTemplate, Settings

import config
from common import get_client, load_index, setup_models


class EduRAG:
    def __init__(self, llm_model: str = config.LLM_MODEL, top_k: int = config.TOP_K):
        setup_models(llm_model)
        self.index = load_index(get_client())
        self.top_k = top_k
        qa_template = PromptTemplate(
            config.QA_PROMPT,
            template_var_mappings={"query_str": "question", "context_str": "context"},
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=top_k, text_qa_template=qa_template
        )

    def answer(self, question: str) -> dict:
        resp = self.query_engine.query(question)
        return {
            "question": question,
            "answer": str(resp.response).strip(),
            "contexts": [n.node.get_content() for n in resp.source_nodes],
            "sources": [
                {"score": round(n.score or 0.0, 4), **n.node.metadata}
                for n in resp.source_nodes
            ],
        }

    def generate_questions(self, topic: str, n: int = 5, difficulty: str = "medium",
                           qtype: str = "mcq", k: int = 3) -> list[dict]:
        retrieved = self.index.as_retriever(similarity_top_k=k).retrieve(topic)
        context = "\n\n---\n\n".join(r.node.get_content() for r in retrieved)
        prompt = config.QGEN_PROMPT.format(
            n=n,
            qtype="multiple-choice" if qtype == "mcq" else "short-answer",
            difficulty=difficulty,
            topic=topic,
            format_hint=config.MCQ_FORMAT if qtype == "mcq" else config.SHORT_FORMAT,
            context=context,
        )
        raw = Settings.llm.complete(prompt).text
        return _parse_json_list(raw)


def _parse_json_list(raw: str) -> list[dict]:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"LLM did not return valid JSON:\n{raw}")
