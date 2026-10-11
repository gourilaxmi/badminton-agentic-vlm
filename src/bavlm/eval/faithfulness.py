"""Counterfactual faithfulness -- the strongest contribution.

Necessity alone is not enough: removing any evidence degrades performance a
little, so a raw flip rate says little. The control is what makes the number
mean something.

    faithfulness = flip_rate(cited) - flip_rate(random uncited)

A gap near zero means the citations are arbitrary: the explanation points at
nothing in particular, even if it reads well.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from bavlm.data.schema import Evidence, Trace

# A rerun function takes the evidence set actually shown to the model and
# returns the answer string. Injected so that this module stays testable.
RerunFn = Callable[[Trace, Sequence[Evidence]], str]


@dataclass
class FaithfulnessResult:
    n_questions: int
    cited_flip_rate: float
    random_flip_rate: float
    sufficiency_rate: float

    @property
    def score(self) -> float:
        return self.cited_flip_rate - self.random_flip_rate

    def __str__(self) -> str:
        return (f"n={self.n_questions}  cited={self.cited_flip_rate:.3f}  "
                f"random={self.random_flip_rate:.3f}  score={self.score:+.3f}  "
                f"sufficiency={self.sufficiency_rate:.3f}")


def _flip_rate(trace: Trace, rerun: RerunFn, drop: Sequence[Evidence]) -> float:
    """Fraction of individually-dropped items that change the answer."""
    if not drop:
        return 0.0
    flips = 0
    for item in drop:
        kept = [e for e in trace.evidence if e.eid != item.eid]
        if rerun(trace, kept) != trace.answer:
            flips += 1
    return flips / len(drop)


def evaluate(traces: Sequence[Trace], rerun: RerunFn, seed: int = 0) -> FaithfulnessResult:
    rng = random.Random(seed)
    cited_rates, random_rates, sufficient = [], [], 0
    scored = 0

    for t in traces:
        cited = t.cited_evidence()
        uncited = t.uncited_evidence()
        if not cited:
            continue                      # nothing claimed, nothing to test
        scored += 1

        cited_rates.append(_flip_rate(t, rerun, cited))

        # control: drop the same NUMBER of uncited items, sampled at random
        k = min(len(cited), len(uncited))
        control = rng.sample(uncited, k) if k else []
        random_rates.append(_flip_rate(t, rerun, control))

        # sufficiency: cited evidence alone, nothing else
        if rerun(t, cited) == t.answer:
            sufficient += 1

    n = max(scored, 1)
    return FaithfulnessResult(
        n_questions=scored,
        cited_flip_rate=sum(cited_rates) / n,
        random_flip_rate=sum(random_rates) / n,
        sufficiency_rate=sufficient / n,
    )
