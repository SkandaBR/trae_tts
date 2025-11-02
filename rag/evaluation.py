"""
RAG KPI evaluation utilities.

This module defines a function `evaluate_rag_kpi` that:
- Computes Mean Reciprocal Rank (MRR) for retrieval against ground-truth verse IDs
- Computes a Faithfulness Score for generated answers using ragas or llamaindex
- Prints final metrics and whether KPIs are met

Dependencies:
- For faithfulness via ragas: `pip install ragas`
- For faithfulness via llamaindex: `pip install llama-index`
- An LLM provider for evaluation (e.g., set OPENAI_API_KEY for OpenAI-backed evaluators)

Example usage:
    from bhagavadgita_rag import BhagavadGitaRAG

    rag = BhagavadGitaRAG("e:/Codebase/review4/trae_tts/rag/bhagavadgita_Chapter_18.json")

    def retriever_fn(query, top_k=5):
        return rag.retrieve(query, top_k=top_k)

    def generator_fn(query, contexts):
        # Simple baseline: return the top context as the "answer"
        return contexts[0] if contexts else ""

    evaluate_rag_kpi(None, retriever_fn, generator_fn)
"""

# Module: evaluation utilities

# Module-level logging and helpers
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import logging


def _default_examples() -> List[Dict[str, Any]]:
    """
    Returns 10 example questions mapped to ground-truth verse IDs in Chapter 18.

    IDs follow the convention used in the Chroma builder: 'ch{chapter}_v{verse}'.
    """
    return [
        {"question": "ತ್ಯಾಗ ಮತ್ತು ಸಂನ್ಯಾಸದ ವ್ಯತ್ಯಾಸವೇನು?", "ground_truth_id": "ch18_v2"},
        {"question": "ಯಜ್ಞ, ದಾನ ಮತ್ತು ತಪಸ್ಸನ್ನು ತ್ಯಜಿಸಬೇಕೆ?", "ground_truth_id": "ch18_v5"},
        {"question": "ತ್ಯಾಗದ ಮೂರು ವಿಧಗಳು ಯಾವುವು?", "ground_truth_id": "ch18_v4"},
        {"question": "ನಿಗದಿತ ಕರ್ಮವನ್ನು ತ್ಯಜಿಸುವುದು ಯೋಗ್ಯವೇ?", "ground_truth_id": "ch18_v7"},
        {"question": "ದುಃಖ ಅಥವಾ ಶ್ರಮದಿಂದ ಕರ್ಮ ತ್ಯಜಿಸುವುದು ಯಾವ ಗುಣ?", "ground_truth_id": "ch18_v8"},
        {"question": "ಕರ್ತವ್ಯವನ್ನು ಆಸಕ್ತಿ ಇಲ್ಲದೆ ಮಾಡುವ ತ್ಯಾಗ ಯಾವುದು?", "ground_truth_id": "ch18_v9"},
        {"question": "ತ್ಯಾಗಿ ಒಳ್ಳೆಯ ಅಥವಾ ಕೆಟ್ಟ ಕರ್ಮವನ್ನು ಹೇಗೆ ನೋಡುವನು?", "ground_truth_id": "ch18_v10"},
        {"question": "ಕರ್ಮಗಳ ಫಲವನ್ನು ತ್ಯಜಿಸುವವನನ್ನು ಯಾರನ್ನೆಂದು ಕರೆಯುತ್ತಾರೆ?", "ground_truth_id": "ch18_v11"},
        {"question": "ಕರ್ಮದ ಫಲಗಳು ತ್ಯಜಿಸದವರಿಗೆ ಯಾವವು?", "ground_truth_id": "ch18_v12"},
        {"question": "ಕರ್ಮಸಾಧನೆಗೆ ಪಂಚ ಕಾರಣ ಯಾವುವು?", "ground_truth_id": "ch18_v14"},
    ]


def _result_to_ids_and_contexts(result: Any) -> Tuple[List[str], List[str]]:
    """
    Normalize retriever outputs to (ranking_ids, contexts).

    Supports:
    - List[{"verse": {...}, "similarity": float}] from BhagavadGitaRAG.retrieve()
    - Chroma `query` style dict with "ids", "documents", "metadatas"
    """
    ranking_ids: List[str] = []
    contexts: List[str] = []

    # BhagavadGitaRAG style
    if isinstance(result, list) and result and isinstance(result[0], dict) and "verse" in result[0]:
        for item in result:
            verse = item.get("verse", {})
            chapter = str(verse.get("chapter", ""))
            verse_num = str(verse.get("verse", ""))
            verse_text = verse.get("text", "")
            translation = verse.get("translation", "")
            english_translation = verse.get("english_translation", "")
            if chapter and verse_num:
                ranking_ids.append(f"ch{chapter}_v{verse_num}")
            # Build a rich context string
            ctx = "\n\n".join(
                [part for part in [verse_text,
                                   f"Translation: {translation}" if translation else "",
                                   f"English: {english_translation}" if english_translation else ""]
                 if part]
            ).strip()
            contexts.append(ctx if ctx else verse_text)
        return ranking_ids, contexts

    # Chroma style
    if isinstance(result, dict) and "ids" in result and "documents" in result:
        ids_nested = result.get("ids", [[]])
        docs_nested = result.get("documents", [[]])
        if ids_nested and isinstance(ids_nested[0], list):
            ranking_ids = [str(x) for x in ids_nested[0]]
        if docs_nested and isinstance(docs_nested[0], list):
            contexts = [str(x) for x in docs_nested[0]]
        return ranking_ids, contexts

    # Unknown shape: best effort
    try:
        # Treat as sequence of strings
        contexts = [str(x) for x in result] if isinstance(result, (list, tuple)) else [str(result)]
    except Exception:
        contexts = [str(result)]
    return ranking_ids, contexts


def _compute_mrr(ground_truth_id: str, ranking_ids: List[str]) -> float:
    try:
        idx = ranking_ids.index(ground_truth_id)
        return 1.0 / (idx + 1)
    except ValueError:
        return 0.0


def _faithfulness_with_ragas(dataset: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute faithfulness using ragas if available.
    Returns the faithfulness score in [0, 1], or None if evaluator is not available.
    """
    logger.debug("Starting RAGAS faithfulness evaluation; dataset size=%d", len(dataset))
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness as ragas_faithfulness
        logger.debug("Imported ragas core modules successfully.")
    except Exception as e_import:
        logger.exception("RAGAS core imports failed: %s", e_import)
        return None

    # Attempt to build a proper RAGAS Dataset across versions
    ragas_ds = None
    try:
        module_ds = __import__("ragas.dataset", fromlist=["Dataset"])
        RagasDataset = getattr(module_ds, "Dataset")

        # Try to locate a compatible Sample class in various ragas modules
        sample_cls = None
        sample_candidates = [
            ("ragas.dataset", "SingleTurnSample"),
            ("ragas.samples", "SingleTurnSample"),
            ("ragas.dataset", "QASample"),
            ("ragas.samples", "QASample"),
            ("ragas.dataset", "Sample"),
            ("ragas.samples", "Sample"),
            ("ragas.schema", "SingleTurnSample"),
            ("ragas.schema", "Sample"),
            ("ragas.types", "SingleTurnSample"),
            ("ragas.types", "Sample"),
        ]
        for mod_name, class_name in sample_candidates:
            try:
                mod = __import__(mod_name, fromlist=[class_name])
                sample_cls = getattr(mod, class_name)
                logger.debug("Using RAGAS sample class %s from %s", class_name, mod_name)
                break
            except Exception:
                continue

        if sample_cls is not None:
            samples = []
            for item in dataset:
                q = str(item.get("question", ""))
                a = str(item.get("answer", ""))
                ctxs = [str(c) for c in item.get("contexts", [])]
                # Try common constructor arg shapes across versions
                ctor_kwargs_options = [
                    {"question": q, "answer": a, "contexts": ctxs},
                    {"user_input": q, "response": a, "retrieved_contexts": ctxs, "ground_truth": None},
                    {"prompt": q, "output": a, "contexts": ctxs},
                ]
                sample_obj = None
                for kw in ctor_kwargs_options:
                    try:
                        sample_obj = sample_cls(**kw)
                        break
                    except Exception:
                        continue
                if sample_obj is None:
                    raise RuntimeError("Failed to instantiate RAGAS sample with provided kwargs")
                samples.append(sample_obj)
            ragas_ds = RagasDataset(items=samples)
            logger.debug("Constructed RAGAS Dataset with %d samples via sample class.", len(samples))
        else:
            # Fall back to factory methods if no sample class was found
            for meth in ("from_list", "from_dict", "from_items"):
                try:
                    factory = getattr(RagasDataset, meth)
                    ragas_ds = factory(dataset)
                    logger.debug("Constructed RAGAS Dataset via %s.", meth)
                    break
                except Exception:
                    continue

    except Exception as e_ds:
        logger.exception("Failed to construct RAGAS Dataset: %s", e_ds)
        ragas_ds = None

    if ragas_ds is None:
        logger.debug("RAGAS Dataset construction unavailable; returning None.")
        return None

    # Evaluate and extract an average faithfulness score robustly
    try:
        res = ragas_evaluate(dataset=ragas_ds, metrics=[ragas_faithfulness])
        logger.debug("RAGAS evaluate executed with Dataset.")
    except Exception as e_eval:
        logger.exception("RAGAS evaluate failed: %s", e_eval)
        return None

    avg_score: Optional[float] = None

    # Try Result.to_pandas() if available
    try:
        to_pandas = getattr(res, "to_pandas", None)
        if callable(to_pandas):
            df = to_pandas()
            cols = getattr(df, "columns", [])
            if "faithfulness" in cols:
                avg_score = float(df["faithfulness"].astype(float).mean())
                logger.debug("RAGAS DataFrame mean faithfulness=%s", avg_score)
    except Exception as e_df:
        logger.debug("RAGAS to_pandas extraction failed: %s", e_df)

    # If res is a pandas DataFrame
    if avg_score is None:
        try:
            import pandas as pd  # type: ignore
            if isinstance(res, pd.DataFrame) and "faithfulness" in res.columns:
                avg_score = float(res["faithfulness"].astype(float).mean())
                logger.debug("RAGAS DataFrame mean faithfulness=%s", avg_score)
        except Exception:
            pass

    # Dict-like result fallbacks
    if avg_score is None and isinstance(res, dict):
        logger.debug("RAGAS result keys: %s", list(res.keys()))
        if isinstance(res.get("faithfulness"), (float, int)):
            avg_score = float(res["faithfulness"])
        metrics = res.get("metrics")
        if avg_score is None and isinstance(metrics, dict):
            val = metrics.get("faithfulness")
            if isinstance(val, (float, int)):
                avg_score = float(val)
        scores = res.get("scores")
        if avg_score is None and isinstance(scores, list) and scores:
            try:
                nums = [float(s) for s in scores if isinstance(s, (int, float))]
                if nums:
                    avg_score = sum(nums) / len(nums)
            except Exception:
                pass
        results = res.get("results")
        if avg_score is None and isinstance(results, list) and results:
            vals: List[float] = []
            for r in results:
                if isinstance(r, dict):
                    for k in ("faithfulness", "score"):
                        v = r.get(k)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
            if vals:
                avg_score = sum(vals) / len(vals)

    if avg_score is not None:
        return avg_score

    logger.debug("RAGAS evaluation returned unsupported type: %s", type(res))
    return None


def _faithfulness_with_llamaindex(dataset: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute faithfulness using llamaindex if available and an LLM is configured.
    Returns average faithfulness score in [0, 1], or None if evaluator is not available.
    """
    logger.debug("Starting LlamaIndex faithfulness evaluation; dataset size=%d", len(dataset))
    try:
        from llama_index.core.evaluation import FaithfulnessEvaluator
        from llama_index.llms.openai import OpenAI as LIOpenAI
        logger.debug("Imported LlamaIndex FaithfulnessEvaluator and OpenAI LLM.")

        api_key = os.environ.get("OPENAI_API_KEY")
        logger.debug("OPENAI_API_KEY present? %s", bool(api_key))
        if not api_key:
            return None

        llm = LIOpenAI(api_key=api_key)
        evaluator = FaithfulnessEvaluator(llm=llm)
        scores: List[float] = []

        for idx, item in enumerate(dataset, start=1):
            q = item["question"]
            a = item["answer"]
            ctxs = item["contexts"]
            logger.debug("Evaluating item %d: q len=%d, contexts=%d", idx, len(q), len(ctxs))
            # Prefer evaluate() with string response
            try:
                resp = evaluator.evaluate(query=q, response=a, contexts=ctxs)
                logger.debug("Used evaluate() for item %d.", idx)
            except TypeError as e_eval_str:
                logger.debug("evaluate() failed for item %d: %s; trying evaluate_response.", idx, e_eval_str)
                # Try multiple Response import paths
                resp_obj = None
                import_error: Optional[Exception] = None
                for path in (
                    "llama_index.core.response.schema",
                    "llama_index.core.schema",
                    "llama_index.core.base.response",
                ):
                    try:
                        module = __import__(path, fromlist=["Response"])
                        Response = getattr(module, "Response")
                        resp_obj = Response(response=a)
                        logger.debug("Imported Response from %s.", path)
                        break
                    except Exception as e_imp:
                        import_error = e_imp
                        continue
                if resp_obj is None:
                    logger.debug("Failed to import Response from known paths: %s", import_error)
                    return None
                resp = evaluator.evaluate_response(query=q, response=resp_obj, contexts=ctxs)
                logger.debug("Used evaluate_response() with Response object for item %d.", idx)

            score = getattr(resp, "score", None)
            passing = getattr(resp, "passing", None)
            logger.debug("Item %d evaluator result: score=%s, passing=%s", idx, score, passing)
            if score is None:
                score = 1.0 if passing else 0.0
            scores.append(float(score))

        avg = (sum(scores) / len(scores)) if scores else None
        logger.debug("LlamaIndex average faithfulness=%s", avg)
        return avg
    except Exception as e_outer:
        logger.exception("LlamaIndex faithfulness flow failed: %s", e_outer)
        return None


def evaluate_rag_kpi(
    queries_and_answers: Optional[List[Dict[str, Any]]],
    retriever: Callable[[str, int], Any],
    generator: Callable[[str, List[str]], str],
    top_k: int = 5,
    mrr_target: float = 0.8,
    faithfulness_target: float = 0.95,
) -> Tuple[float, Optional[float], bool]:
    """
    Evaluate the Bhagavad Gita RAG system against KPIs.

    Inputs:
    - queries_and_answers: List of dicts with keys:
        - 'question': str
        - 'ground_truth_id': str (e.g., 'ch18_v2'); if None, defaults to internal examples
      If None or empty, 10 default examples for Chapter 18 are used.
    - retriever: Callable that accepts (query: str, top_k: int) and returns ranked results.
        Supported outputs:
        - List[{'verse': {...}, 'similarity': float}] as in BhagavadGitaRAG.retrieve()
        - Chroma-style dict {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
    - generator: Callable that accepts (query: str, contexts: List[str]) and returns an answer string.

    Outputs:
    - (mrr, faithfulness, meets_kpi):
        - mrr: float in [0,1] Mean Reciprocal Rank over all queries
        - faithfulness: Optional[float] in [0,1] average faithfulness score; None if evaluation unavailable
        - meets_kpi: bool True iff mrr >= mrr_target and faithfulness >= faithfulness_target

    Side-effects:
    - Prints the final MRR and Faithfulness Score and whether KPIs are met.
    """
    logger.info("Starting KPI evaluation: examples=%s, top_k=%d", "default" if not queries_and_answers else "custom", top_k)
    examples = queries_and_answers or _default_examples()
    if not examples:
        raise ValueError("No queries provided and default examples are unavailable.")

    rranks: List[float] = []
    dataset: List[Dict[str, Any]] = []

    for i, ex in enumerate(examples, start=1):
        q = ex["question"]
        gt_id = ex["ground_truth_id"]
        logger.debug("Query %d: '%s' | Ground truth ID=%s", i, q, gt_id)

        result = retriever(q, top_k=top_k)
        ranking_ids, contexts = _result_to_ids_and_contexts(result)
        logger.debug("Query %d: ranking_ids=%s", i, ranking_ids)

        rr = _compute_mrr(gt_id, ranking_ids)
        rranks.append(rr)
        logger.debug("Query %d: reciprocal_rank=%.4f (top_k=%d)", i, rr, top_k)

        answer = generator(q, contexts)
        dataset.append({"question": q, "answer": answer, "contexts": contexts})
        logger.debug("Query %d: contexts_len=%d, answer_len=%d", i, len(contexts), len(answer))

    mrr = sum(rranks) / len(rranks)
    logger.info("Computed MRR=%.4f over %d queries", mrr, len(examples))

    api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    faithfulness: Optional[float] = None

    # Prefer LlamaIndex if a key is present; otherwise skip straight to RAGAS
    if api_key_present:
        faithfulness = _faithfulness_with_llamaindex(dataset)
        if faithfulness is None:
            logger.debug("LlamaIndex faithfulness unavailable or failed; falling back to RAGAS.")
    if faithfulness is None:
        faithfulness = _faithfulness_with_ragas(dataset)

    meets_kpi = (mrr >= mrr_target) and (faithfulness is not None) and (faithfulness >= faithfulness_target)
    logger.info("Final metrics: MRR=%.4f, Faithfulness=%s, Meets KPI=%s", mrr, f"{faithfulness:.4f}" if faithfulness is not None else "None", meets_kpi)

    print(f"MRR: {mrr:.4f}")
    if faithfulness is None:
        reason = "ragas/llama-index not available or failed"
        print(f"Faithfulness: unavailable ({reason})")
    else:
        print(f"Faithfulness: {faithfulness:.4f}")
    print(f"KPI Met: {'YES' if meets_kpi else 'NO'} (targets: MRR >= {mrr_target}, Faithfulness >= {faithfulness_target})")

    return mrr, faithfulness, meets_kpi


# Add logger setup
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("rag.evaluation")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "evaluation.log"), encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)