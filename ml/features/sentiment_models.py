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
            Alternative models: "yiyanghkust/finbert-tone", "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
            
        Raises:
            ImportError: If transformers library is not installed
            OSError: If model cannot be loaded (network issue, wrong name, or auth required)
        """
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for LLMSentimentModel. Install with `pip install transformers`."
            ) from exc

        # Try to load the model with better error handling
        try:
            self._pipeline = pipeline("text-classification", model=model_name, return_all_scores=True)
            self.model_name = model_name
        except OSError as exc:
            # Provide helpful error message with alternatives
            error_msg = (
                f"Failed to load model '{model_name}'. "
                f"Error: {exc}\n\n"
                "Possible solutions:\n"
                "1. Check your internet connection\n"
                "2. Verify the model name is correct (e.g., 'ProsusAI/finbert')\n"
                "3. If it's a private repo, authenticate with: `huggingface-cli login`\n"
                "4. Try alternative models:\n"
                "   - 'yiyanghkust/finbert-tone'\n"
                "   - 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'\n"
                "5. Install huggingface_hub: `pip install huggingface_hub`"
            )
            raise OSError(error_msg) from exc

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
    use_title: bool = False,
    combine_title_description: bool = True,
) -> pd.DataFrame:
    """
    Combine sentiment scores from multiple sources into unified scores.
    
    Args:
        base_df: DataFrame with text column(s) and optional vendor sentiment
        llm_model: Optional LLM sentiment model
        rule_model: Optional rule-based sentiment model (defaults to RuleBasedSentimentModel())
        text_column: Name of column containing text to analyze (default: "description")
        use_title: If True, use "title" column instead of text_column (default: False)
        combine_title_description: If True and both title/description exist, combine them (default: True)
            This is often better for sentiment as titles are attention-grabbing but descriptions provide context.
            Takes precedence over use_title and text_column.
        
    Returns:
        DataFrame with combined sentiment_score and sentiment_label columns
        
    Note:
        Best practice: Use combine_title_description=True to get both the attention-grabbing
        sentiment from titles and the contextual nuance from descriptions.
    """
    df = base_df.copy()

    # Determine which text column(s) to use
    if combine_title_description and "title" in df.columns and "description" in df.columns:
        # Combine title and description (often best for sentiment)
        df["_combined_text"] = (
            df["title"].fillna("").astype(str) + ". " + df["description"].fillna("").astype(str)
        )
        actual_text_column = "_combined_text"
    elif use_title and "title" in df.columns:
        actual_text_column = "title"
    else:
        actual_text_column = text_column

    if rule_model is None:
        rule_model = RuleBasedSentimentModel()
    try:
        rule_scores = rule_model.score(df, text_column=actual_text_column)
    except ValueError as exc:
        warnings.warn(str(exc))
        rule_scores = pd.DataFrame(index=df.index)

    if llm_model is not None:
        try:
            llm_scores = llm_model.score(df, text_column=actual_text_column)
        except Exception as exc:
            warnings.warn(f"LLM sentiment scoring failed: {exc}. Continuing without LLM scores.")
            llm_scores = pd.DataFrame(index=df.index)
    else:
        llm_scores = pd.DataFrame(index=df.index)
    
    # Clean up temporary column if created
    if "_combined_text" in df.columns:
        df = df.drop(columns=["_combined_text"])

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
        # Fix: Use fillna instead of setdefault to handle existing columns with NaN values
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        else:
            df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
        
        if "sentiment_label" not in df.columns:
            df["sentiment_label"] = "neutral"
        else:
            df["sentiment_label"] = df["sentiment_label"].fillna("neutral")

    return df

