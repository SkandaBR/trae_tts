#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict
from evaluation import evaluate_rag_kpi
from bhagavadgita_rag import BhagavadGitaRAG

def load_properties(path: str) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs from a properties file."""
    props: Dict[str, str] = {}
    if not os.path.exists(path):
        return props
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                props[key.strip()] = val.strip()
    return props

def tail_log(log_path: str, max_lines: int = 100) -> None:
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print("\n=== Evaluation Debug Log (last {} lines) ===".format(max_lines))
        for line in lines[-max_lines:]:
            print(line.rstrip())
    except Exception as e:
        print(f"Could not read log file {log_path}: {e}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load secrets from properties if present (kept out of VCS via .gitignore)
    secrets_path = os.path.join(current_dir, "secrets.properties")
    secrets = load_properties(secrets_path)
    for k, v in secrets.items():
        # Do not overwrite already-set env vars
        if k not in os.environ and v:
            os.environ[k] = v

    # Initialize RAG with Chapter 18 data
    json_path = os.path.join(current_dir, "bhagavadgita_Chapter_18.json")
    rag = BhagavadGitaRAG(json_path)

    # Wire retriever and a simple generator
    def retriever_fn(query: str, top_k: int = 5):
        return rag.retrieve(query, top_k=top_k)

    def generator_fn(query: str, contexts):
        # Baseline generator: just returns the top context as the "answer"
        return contexts[0] if contexts else ""

    # Evaluate against defaults (10 Chapter 18 questions)
    mrr, faithfulness, meets_kpi = evaluate_rag_kpi(
        queries_and_answers=None,
        retriever=retriever_fn,
        generator=generator_fn,
        top_k=5,
        mrr_target=0.8,
        faithfulness_target=0.95,
    )

    # Tail logs to aid analysis in subsequent runs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, "logs", "evaluation.log")
    tail_log(log_path, max_lines=120)

    import sys
    sys.exit(0 if meets_kpi else 1)

if __name__ == "__main__":
    main()