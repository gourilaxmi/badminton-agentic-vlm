# Hand-check: 20 answers from e06 against the key

Deterministic pick (seed 0), 5 per domain, wrong predictions first, priority sub-tasks first (detailed hitting count, Position).

For each row: open the clip, read the question, decide **is the gold label right?** — then record verdict (gold OK / gold WRONG / question bad) and any note. Clips: `https://huggingface.co/datasets/Moujuruo/FineBadmintonBenchmark/resolve/main/finebadmintonbenchmark/hf_video_clips_qa/<clip_id>.mp4`.

| # | domain / subtask | qid | question | gold | pred | conf | verdict / notes |
|---|---|---|---|---|---|---|---|
| 1 | count / detailed hitting count | `select_detail_count_questions/video_002285` | In the provided video segment, how many times was the 'Net lift' performed? A net lift is an underhand shot... | C | D | -0.12 | |
| 2 | count / detailed hitting count | `select_detail_count_questions/video_002357` | In the provided video segment, how many times was the 'Drop shot' performed? A drop shot is typically hit f... | C | D | -0.03 | |
| 3 | count / detailed hitting count | `select_detail_count_questions/video_002391` | In the provided video segment, how many times was the 'Block' performed? A block is a defensive shot played... | C | B | -0.02 | |
| 4 | count / hitting count | `hitcount/video_000709` | In this badminton rally, how many strokes did both players hit in total? | C | C | -0.50 | |
| 5 | count / hitting count | `hitcount/video_000790` | In this badminton rally, how many strokes did the bottom player hit in total? | B | D | -0.52 | |
| 6 | action / action classification | `action_classification/video_000115` | Analyze the action of the player's (at the top) last hit and determine which type of shot they most likely ... | D | C | -0.26 | |
| 7 | action / action localization | `action_localization/video_000339` | In the video segment, which hit number corresponds to Clear: A clear is a shot hit deep into the opponent's... | C | B | -1.13 | |
| 8 | action / action prediction | `action_prediction/video_000417` | After the player at the top makes their final hit, predict the most probable return from the opponent at th... | D | C | -0.49 | |
| 9 | action / action prediction | `action_prediction/video_000474` | What is the most likely outcome of the return from the opponent at the bottom after the player at the top m... | B | B | -0.49 | |
| 10 | action / action prediction | `action_prediction/video_000618` | What is the most likely response from the opponent at the top after the final hit by the player at the bott... | D | C | -0.50 | |
| 11 | position / hitting localization | `hitting_localization_new/video_001172` | Choose the closest option: Which area of their court (facing direction as front) did the player at the bott... | C | B | -1.04 | |
| 12 | position / hitting localization | `hitting_localization_new/video_001208` | Which area of their own court (with their facing direction as front) did the player at the top execute thei... | C | C | -0.52 | |
| 13 | position / moving recognition | `moving_recognization/video_001551` | Which option corresponds to the player's (at the top) movement relative to their facing direction during th... | B | C | -0.20 | |
| 14 | position / moving recognition | `moving_recognization/video_001559` | What option describes the movement direction of the player (at the top) relative to their facing direction ... | D | A | -0.78 | |
| 15 | position / moving recognition | `moving_recognization/video_001619` | Which option best aligns with the player's (at the top) movement relative to their facing direction during ... | A | B | -0.81 | |
| 16 | cognition / hitting comment | `hitting_comment2_updated/video_000941` | For the last shot by the player at the bottom of the screen, which of the following evaluations best descri... | B | B | -0.00 | |
| 17 | cognition / hitting comment | `hitting_comment2_updated/video_001055` | For the last shot of the player on top of the screen, which of the following evaluations is the most approp... | D | A | -0.70 | |
| 18 | cognition / hitting comment | `hitting_comment2_updated/video_001057` | Based on the action and intention of the top player's last shot in the clip, select the most appropriate ev... | A | B | -1.18 | |
| 19 | cognition / hitting comment | `hitting_comment2_updated/video_001099` | Regarding the last shot executed by the player at the top of the screen, which of the following comments is... | B | C | -0.84 | |
| 20 | cognition / point reason | `pointer_recog_updated/video_001742` | Please analyze the main reason for scoring in this rally. | B | A | -0.11 | |

## Summary

- gold wrong: 
- question unusable: 
- model error confirmed: 
- surprising (model wrong, gold arguable): 
