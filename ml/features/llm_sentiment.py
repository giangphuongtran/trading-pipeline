"""Optional LLM-based sentiment scoring for news."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

import pandas as pd

from .sentiment_models import RuleBasedSentimentModel


@dataclass
class LLMSentimentConfig:
    """Configuration for LLM sentiment scoring."""

    provider: str = "finbert"  # mock | finbert | other
    text_column: str = "description"
    use_title: bool = True
    combine_title_description: bool = True
    batch_size: int = 16


def _build_text_series(df: pd.DataFrame, config: LLMSentimentConfig) -> pd.Series:
    if config.combine_title_description and "title" in df.columns and "description" in df.columns:
        combined = df["title"].fillna("").astype(str) + ". " + df["description"].fillna("").astype(str)
        return combined
    if config.use_title and "title" in df.columns:
        return df["title"].fillna("").astype(str)
    return df.get(config.text_column, pd.Series([""] * len(df))).fillna("").astype(str)


@lru_cache(maxsize=1)
def _get_finbert_pipeline():
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for provider='finbert'. "
            "Install with: pip install transformers torch"
        ) from exc

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
    )


def _score_finbert(texts: Iterable[str], batch_size: int) -> List[dict]:
    pipe = _get_finbert_pipeline()
    results: List[dict] = []
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            results.extend(pipe(batch, return_all_scores=True))
            batch = []
    if batch:
        results.extend(pipe(batch, return_all_scores=True))
    return results


def _llm_score_to_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def add_llm_sentiment(news_df: pd.DataFrame, config: LLMSentimentConfig) -> pd.DataFrame:
    """
    Add sentiment_llm_score and sentiment_llm_label columns to news_df.

    The default provider ("mock") uses the rule-based model as a stand-in.
    """
    if news_df.empty:
        return news_df

    provider = (config.provider or "mock").lower()
    df = news_df.copy()

    if provider == "mock":
        model = RuleBasedSentimentModel()
        text_series = _build_text_series(df, config)
        scores = model.score(df.assign(_text=text_series), text_column="_text")
        df["sentiment_llm_score"] = scores["sentiment_rule_score"]
        df["sentiment_llm_label"] = scores["sentiment_rule_label"]
        return df
    if provider == "finbert":
        text_series = _build_text_series(df, config)
        try:
            raw = _score_finbert(text_series.tolist(), config.batch_size)
        except Exception:
            # Fallback to mock if transformers/torch not available
            model = RuleBasedSentimentModel()
            scores = model.score(df.assign(_text=text_series), text_column="_text")
            df["sentiment_llm_score"] = scores["sentiment_rule_score"]
            df["sentiment_llm_label"] = scores["sentiment_rule_label"]
            return df
        scores: list[float] = []
        labels: list[str] = []
        for row in raw:
            # row is list[dict(label, score)]
            pos = neg = neu = 0.0
            for entry in row:
                label = str(entry.get("label", "")).lower()
                score = float(entry.get("score", 0.0))
                if "pos" in label:
                    pos = score
                elif "neg" in label:
                    neg = score
                elif "neu" in label:
                    neu = score
            llm_score = pos - neg
            scores.append(llm_score)
            labels.append(_llm_score_to_label(llm_score))
        df["sentiment_llm_score"] = scores
        df["sentiment_llm_label"] = labels
        return df

    raise RuntimeError(
        f"LLM provider '{config.provider}' is not configured in this environment. "
        "Use provider='mock' or implement the provider integration."
    )
