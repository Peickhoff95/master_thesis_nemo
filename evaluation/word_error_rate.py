#---
#jupyter:
#   jupytext:
#       text_representation:
#           extension: .py
#           format_name: percent
#           format_version: '1.3'
#           jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from typing import List, Tuple
import editdistance
import re
import pandas as pd

def clean_transcript(transcript: str) -> str:
    # Any sequence of at least two whitespaces
    pattern = r'\\ {2,}'
    transcript = re.sub(pattern, ' ', transcript)
    transcript = transcript.strip()
    return transcript

def calculate_worddist_and_wordlen(truth: str,
                                hyp: str
                               ) -> Tuple[float, float]:
    truth_words: List[str] = truth.split(' ')
    hyp_words: List[str] = hyp.split(' ')

    dist = editdistance.distance(truth_words, hyp_words)
    wordlen = len(truth_words)
    return dist, wordlen

def calculate_wer(truth: str,
                   hyp: str
                  ) -> float:
    dist, wordlen = calculate_worddist_and_wordlen(truth, hyp)
    return dist/wordlen

def calculate_wer_from_series(series: pd.Series,
                               column_truth: str,
                               column_hyp: str
                              ) -> float:
    truth: str = series[column_truth]
    hyp: str = series[column_hyp]
    return calculate_wer(truth, hyp)

def calculate_wer_from_list(truths: List,
                               hypotheses: List,
                              ) -> float:
    truth: str = ','.join(truths)
    hyp: str = ','.join(hypotheses)
    return calculate_wer(truth, hyp)
