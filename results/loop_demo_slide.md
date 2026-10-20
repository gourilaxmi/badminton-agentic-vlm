# Loop demo trace -- qualitative, mock agent, no accuracy claimed

**Question (cognition):** For the last shot of the bottom player in this rally, which of the following evaluations best matches the actual performance?  
**qid** `hitting_comment2_updated/video_000851` · **clip** `video_000851` · **frames used** 8 · **rounds** 2

*frame cache: data\frame_cache*

### Round 1
**Plan** -- seek `the stroke played, coarse, across the rally`  
spans [0.0-3.0] @ 8.33 fps · resolution low · crop full · budget 25 frames

**Observer** -- 6 frames shown, evidence:
- `e1` [0.00-3.00s] (right back court) rally ends with the shuttle landing in court

**Reflection** -- **insufficient** (confidence 0.55): evidence covers only the opening; seek the later part of the rally at higher density

### Round 2
**Plan** -- seek `the racket-shuttle contact moment`  
spans [1.2-2.1] @ 25.0 fps · resolution high · crop full · budget 25 frames

**Observer** -- 2 frames shown, evidence:
- `e2` [0.55-0.95s] (right mid court) <net shot> racket meets the shuttle, contact 1 visible
- `e3` [1.30-1.70s] (right front court) <drop> racket meets the shuttle, contact 2 visible
- `e4` [2.05-2.45s] (mid back court) <smash> racket meets the shuttle, contact 3 visible

**Reflection** -- sufficient (confidence 0.82)

### Answer -- B
- cites: `e3`, `e4`
- explanation: The answer follows from e3, e4: the round-2 observations cover the window the first round missed.
