"""Loop orchestration. Max 3 rounds; writes a full Trace per question.

AVP found gains saturate at three rounds -- do not raise the cap without
an ablation showing it helps.

The loop is deliberately backend-agnostic: planner, observer and reflector all
call `backend.generate`, so the same loop runs against the mock (this sprint's
demo), Qwen on Kaggle or Gemini -- comparisons are then of observation strategy,
not of plumbing.

Review-2 scope (HANDOVER step 5): the demo is QUALITATIVE. Traces are the
deliverable; no accuracy is claimed. The training-free evaluation against the
baselines is roadmap phase 3 (`scripts/03_run_agent.py`).

Budget note for phase 3: each round may use up to `budget` frames (that is how
prompts/planner.txt phrases it), so a 2-round trace can show up to 2* budget
frames in total. Any comparison against the uniform baseline must be made at
equal TOTAL frames (rule 4) -- decide the total-budget split before phase 3
runs, and record `Trace.frames_used` either way.
"""
from __future__ import annotations

import time
from dataclasses import asdict

from bavlm.agent import observer, planner, reflector
from bavlm.data.schema import Trace

MAX_ROUNDS = 3
DEFAULT_DURATION = 10.0  # seconds, used only when no frames are on disk:
                         # FBBench clips span roughly 2-12 s under the cap


class AgentLoop:
    def __init__(self, backend, budget: int = 25, max_rounds: int = MAX_ROUNDS,
                 clip_fps: float = 2.0) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self.backend = backend
        self.budget = budget
        self.max_rounds = max_rounds
        self.clip_fps = clip_fps

    # ------------------------------------------------------------------ calls
    def _gen(self, frames, prompt: str) -> str:
        out = self.backend.generate(frames, prompt)
        stats = getattr(self.backend, "last_stats", None) or {}
        self._tokens += int(stats.get("input_tokens") or 0)
        return out

    # ------------------------------------------------------------------- run
    def run(self, *, qid: str, clip_id: str, question: str, frames: list[str] | tuple = (),
            domain: str | None = None, duration: float | None = None) -> Trace:
        """One question end to end: plan -> observe -> reflect -> (re-plan)*.

        `duration` is the clip length in seconds; without frames it falls back
        to DEFAULT_DURATION and the caller should say so -- a planner shown the
        wrong rally length plans against it.
        """
        trace = Trace(question_id=qid, clip_id=clip_id, question=question, domain=domain)
        t0 = time.time()
        self._tokens = 0
        frames = list(frames)
        dur = (float(duration) if duration
               else (len(frames) / self.clip_fps if frames else DEFAULT_DURATION))

        history = ""
        eid = 0
        final: reflector.Judgement | None = None
        for rnd in range(1, self.max_rounds + 1):
            raw = self._gen([], planner.build_prompt(question, dur, self.budget, history))
            plan = planner.parse(raw, self.budget)
            shown, found = observer.execute(plan, frames, self.backend, cache_fps=self.clip_fps)
            for e in found:
                eid += 1
                e.eid = f"e{eid}"
                trace.evidence.append(e)
            trace.frames_used += len(shown)

            lines = [f"{e.eid}: {e.to_prompt_line()}" for e in trace.evidence]
            j = reflector.parse(self._gen([], reflector.build_prompt(
                question, lines, rnd, self.max_rounds)))

            known = {e.eid for e in trace.evidence}
            dropped = [c for c in j.cites if c not in known]
            if dropped:
                j.cites = [c for c in j.cites if c in known]
            trace.rounds.append({
                "round": rnd,
                "plan": asdict(plan),
                "frames_this_round": len(shown),
                "evidence_ids": [e.eid for e in found],
                "sufficient": j.sufficient,
                "confidence": j.confidence,
                "missing": "" if j.sufficient else j.missing,
                **({"dropped_cites": dropped} if dropped else {}),
            })
            final = j
            if j.sufficient:
                break
            history += f"Round {rnd}: sought '{plan.what}'"
            if j.missing:
                history += f"; observer reported missing: {j.missing}"
            history += "\n"

        if final is not None:
            trace.answer = final.answer
            trace.cited = list(final.cites)
            trace.explanation = final.explanation
        trace.input_tokens = self._tokens
        trace.wall_seconds = round(time.time() - t0, 3)
        return trace
