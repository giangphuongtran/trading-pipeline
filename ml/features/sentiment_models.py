"""Sentiment modelling utilities for news features."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


POSITIVE_WORDS = {
    "beat",
    "growth",
    "bullish",
    "surge",
    "optimistic",
    "outperform",
    "rally",
    "upgrade",
    "profit",
    "record",
    "strong",
    "positive",
}

NEGATIVE_WORDS = {
    "miss",
    "loss",
    "bearish",
    "drop",
    "pessimistic",
    "underperform",
    "selloff",
    "downgrade",
    "decline",
    "weak",
    "negative",
    "lawsuit",
}


@dataclass
class RuleBasedSentimentModel:
    """Lightweight lexicon-based sentiment scorer."""

    positive_words: Iterable[str] = field(default_factory=lambda: set(POSITIVE_WORDS))
    negative_words: Iterable[str] = field(default_factory=lambda: set(NEGATIVE_WORDS))

    def score(self, df: pd.DataFrame, text_column: str = "description") -> pd.DataFrame:
        """
        Score sentiment using keyword matching.
        
        Args:
            df: DataFrame with text column
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with sentiment_rule_score and sentiment_rule_label columns
        """
        if text_column not in df.columns:
            raise ValueError(f"RuleBasedSentimentModel expects column '{text_column}'")

        pos = set(w.lower() for w in self.positive_words)
        neg = set(w.lower() for w in self.negative_words)

        def _score(text: str) -> float:
            if not isinstance(text, str):
                return 0.0
            tokens = text.lower().split()
            if not tokens:
                return 0.0
            score = 0
            for token in tokens:
                if token in pos:
                    score += 1
                elif token in neg:
                    score -= 1
            return score / math.sqrt(len(tokens))

        scores = df[text_column].apply(_score)
        labels = np.where(scores > 0.1, "positive", np.where(scores < -0.1, "negative", "neutral"))

        return pd.DataFrame(
            {
                "sentiment_rule_score": scores,
                "sentiment_rule_label": labels,
            },
            index=df.index,
        )


class LLMSentimentModel:
    """Wrapper around transformer-based sentiment classifiers."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize LLM sentiment model.
        
        Args:
            model_name: HuggingFace model name (default: "ProsusAI/finbert")
            
        Raises:
            ImportError: If transformers library is not installed
        """
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for LLMSentimentModel. Install with `pip install transformers`."
            ) from exc

        self._pipeline = pipeline("text-classification", model=model_name, return_all_scores=True)
        self.model_name = model_name

    def score(self, df: pd.DataFrame, text_column: str = "description") -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"LLMSentimentModel expects column '{text_column}'")

        texts = df[text_column].fillna("").astype(str).tolist()
        if not texts:
            return pd.DataFrame(columns=["sentiment_llm_score", "sentiment_llm_label"], index=df.index)

        outputs = self._pipeline(texts, truncation=True)

        scores = []
        labels = []
        for score_list in outputs:
            label_to_score = {item["label"].lower(): item["score"] for item in score_list}
            pos = label_to_score.get("positive") or label_to_score.get("pos", 0.0)
            neg = label_to_score.get("negative") or label_to_score.get("neg", 0.0)
            neu = label_to_score.get("neutral", 0.0)
            numeric = pos - neg
            if np.isnan(numeric):
                numeric = 0.0
            scores.append(numeric)
            if pos >= max(neg, neu):
                labels.append("positive")
            elif neg >= max(pos, neu):
                labels.append("negative")
            else:
                labels.append("neutral")

        return pd.DataFrame(
            {
                "sentiment_llm_score": scores,
                "sentiment_llm_label": labels,
            },
            index=df.index,
        )


def combine_sentiment_scores(
    base_df: pd.DataFrame,
    *,
    llm_model: Optional[LLMSentimentModel] = None,
    rule_model: Optional[RuleBasedSentimentModel] = None,
    text_column: str = "description",
) -> pd.DataFrame:
    """
    Combine sentiment scores from multiple sources into unified scores.
    
    Args:
        base_df: DataFrame with text column and optional vendor sentiment
        llm_model: Optional LLM sentiment model
        rule_model: Optional rule-based sentiment model (defaults to RuleBasedSentimentModel())
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with combined sentiment_score and sentiment_label columns
    """
    df = base_df.copy()

    if rule_model is None:
        rule_model = RuleBasedSentimentModel()
    try:
        rule_scores = rule_model.score(df, text_column=text_column)
    except ValueError as exc:
        warnings.warn(str(exc))
        rule_scores = pd.DataFrame(index=df.index)

    if llm_model is not None:
        llm_scores = llm_model.score(df, text_column=text_column)
    else:
        llm_scores = pd.DataFrame(index=df.index)

    for col in rule_scores.columns:
        df[col] = rule_scores[col]
    for col in llm_scores.columns:
        df[col] = llm_scores[col]

    available_scores = []
    if "sentiment_llm_score" in df.columns:
        available_scores.append(df["sentiment_llm_score"])
    if "sentiment_rule_score" in df.columns:
        available_scores.append(df["sentiment_rule_score"])
    if "sentiment_score" in df.columns:
        df["sentiment_vendor_score"] = df["sentiment_score"]
        available_scores.append(df["sentiment_vendor_score"])

    if available_scores:
        combined = pd.concat(available_scores, axis=1).mean(axis=1)
        df["sentiment_score"] = combined
        df["sentiment_label"] = np.where(
            combined > 0.1, "positive", np.where(combined < -0.1, "negative", "neutral")
        )
    else:
        df.setdefault("sentiment_score", 0.0)
        df.setdefault("sentiment_label", "neutral")

    return df

