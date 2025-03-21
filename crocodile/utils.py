from typing import Dict, List, Literal, Set, TypedDict

from nltk.tokenize import word_tokenize

from crocodile import STOP_WORDS

LitType = Literal["NUMBER", "STRING", "DATETIME"]
NerType = Literal["LOCATION", "ORGANIZATION", "PERSON", "OTHER"]


class ColType(TypedDict):
    NE: Dict[str, LitType | NerType]
    LIT: Dict[str, LitType | NerType]
    IGNORED: List[str]


def ngrams(string: str, n: int = 3) -> List[str]:
    tokens: List[str] = [string[i : i + n] for i in range(len(string) - n + 1)]
    return tokens


def tokenize_text(text: str) -> Set[str]:
    tokens: List[str] = word_tokenize(text.lower())
    return {t for t in tokens if t not in STOP_WORDS}
