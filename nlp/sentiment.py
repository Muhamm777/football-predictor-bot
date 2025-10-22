from typing import List, Dict

# Placeholder sentiment scoring (-1..1)
def sentiment_score(texts: List[str]) -> List[float]:
    scores: List[float] = []
    for t in texts:
        s = 0.0
        tl = t.lower()
        negative = [
            "плохо", "ужасно", "слабый", "провал", "проигрыш", "слит", "bad", "awful", "weak", "lose"
        ]
        positive = [
            "хорошо", "отлично", "сильный", "победа", "тащит", "win", "great", "strong", "good"
        ]
        if any(w in tl for w in negative):
            s -= 0.5
        if any(w in tl for w in positive):
            s += 0.5
        scores.append(max(-1.0, min(1.0, s)))
    return scores
