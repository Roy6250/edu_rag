"""CLI entry point.

  python app.py ingest   --pdf_dir data/ --subject Physics
  python app.py ask      "How can acceleration be determined using a velocity-time graph?"
  python app.py generate --topic kinematics --n 5 --difficulty medium --type mcq
  python app.py eval     --csv eval_set.csv --llm gpt-4o-mini
"""
import argparse
import json

import config


def main() -> None:
    p = argparse.ArgumentParser(description="RAG question answering & generation for education")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--pdf_dir", default="data")
    s.add_argument("--subject", required=True)

    s = sub.add_parser("ask")
    s.add_argument("question")
    s.add_argument("--llm", default=config.LLM_MODEL)
    s.add_argument("--top_k", type=int, default=config.TOP_K)

    s = sub.add_parser("generate")
    s.add_argument("--topic", required=True)
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    s.add_argument("--type", dest="qtype", choices=["mcq", "short"], default="mcq")
    s.add_argument("--llm", default=config.LLM_MODEL)

    s = sub.add_parser("eval")
    s.add_argument("--csv", required=True, help="columns: question, ground_truth")
    s.add_argument("--llm", default=config.LLM_MODEL)
    s.add_argument("--out", help="optional per-question scores CSV")

    args = p.parse_args()

    # Imports are deferred so each command only loads what it needs.
    if args.cmd == "ingest":
        from ingest import ingest
        ingest(args.pdf_dir, args.subject)
    elif args.cmd == "ask":
        from rag import EduRAG
        out = EduRAG(llm_model=args.llm, top_k=args.top_k).answer(args.question)
        print(f"\nAnswer: {out['answer']}\n\nSources:")
        for src in out["sources"]:
            print(f"  - {src}")
    elif args.cmd == "generate":
        from rag import EduRAG
        qs = EduRAG(llm_model=args.llm).generate_questions(
            args.topic, n=args.n, difficulty=args.difficulty, qtype=args.qtype
        )
        print(json.dumps(qs, indent=2, ensure_ascii=False))
    elif args.cmd == "eval":
        from evaluate import run_eval
        run_eval(args.csv, llm_model=args.llm, out_csv=args.out)


if __name__ == "__main__":
    main()
