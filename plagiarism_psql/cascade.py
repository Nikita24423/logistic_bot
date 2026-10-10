"""
Каскадная классификация заимствований — ЕДИНСТВЕННЫЙ источник истины.

Ранее типы заимствований (exact/paraphrase/semantic) считались отдельно
на TypeScript в guard-main и отдельно здесь — с разными порогами. Теперь
классификация выполняется только в этом модуле; веб-платформа использует
`match_type` из ответа ML-сервиса, а её локальный каскад — офлайн-фолбэк
на случай недоступности ML.

Каскад по источнику (первый сработавший тир определяет тип):
  1. exact      — лексический косинус ≥ EXACT_THRESHOLD
                  (дословное копирование: совпали конкретные слова).
                  Порог отдельный от LEXICAL_SCORE_THRESHOLD, по которому
                  ведётся лексический ПОИСК: при совпадении этих порогов тир
                  вырождается, потому что поиск и так возвращает только хиты
                  выше своего порога, и `exact` получал бы каждый из них;
  2. paraphrase — совпадение 3-грамм по смысловым словам (Жаккар)
                  ≥ PARAPHRASE_THRESHOLD (переформулировка с сохранением
                  смысловой лексики);
  3. semantic   — только семантический сигнал dense-эмбеддингов
                  (глубокий рерайт: смысл совпал, слова — нет).
"""

from __future__ import annotations

import re

# Стоп-слова: служебные слова несут мало сигнала о заимствовании,
# но сильно завышают сходство несвязанных текстов (ru + en).
STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все всё она так его но да ты к у же
    вы за бы по только ее её мне было вот от меня еще ещё нет о из ему когда
    даже ну ли если уже или ни быть был до вас вам ведь там потом себя может
    они тут где есть надо для мы тебя их чем была сам без чего раз тоже себе
    под будет тогда кто этот того потому этого какой ним здесь этом один мой
    тем чтобы нее неё были куда всех при об хотя тот через эти нас про всего
    них эту это той им более также
    the a an and or of to in on for with is are was were be been by at from
    as it its this that these those not no but if then than such can may
    """.split()
)

EXACT = "exact"
PARAPHRASE = "paraphrase"
SEMANTIC = "semantic"

# Порог Жаккара 3-грамм смысловых слов для тира «перефраз»
# (соответствует бывшему PARAPHRASE_THRESHOLD=15% в guard-main).
DEFAULT_PARAPHRASE_THRESHOLD = 0.15


def content_tokens(text: str) -> list[str]:
    """Смысловые слова: без пунктуации, коротких слов и стоп-слов."""
    return [
        t
        for t in re.findall(r"[a-zа-яё0-9]+", (text or "").lower())
        if len(t) > 2 and t not in STOPWORDS
    ]


def paraphrase_similarity(text1: str, text2: str, k: int = 3) -> float:
    """
    Оценка перефразирования: симметричный Жаккар k-грамм по смысловым словам.
    Ловит переписанный текст без дословных совпадений. Возвращает 0..1.
    """

    def build(tokens: list[str]) -> set[str]:
        if len(tokens) < k:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}

    a = build(content_tokens(text1))
    b = build(content_tokens(text2))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def classify_match(
    max_lexical_score: float,
    paraphrase_score: float,
    exact_threshold: float,
    paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD,
) -> str:
    """Тип заимствования источника по каскаду exact → paraphrase → semantic.

    `exact_threshold` — порог тира «дословное копирование», НЕ порог лексического
    поиска: см. модульный docstring."""
    if max_lexical_score >= exact_threshold:
        return EXACT
    if paraphrase_score >= paraphrase_threshold:
        return PARAPHRASE
    return SEMANTIC
