# Setting up Claude Code for the BTP

Terminal CLI on your laptop. ~15 minutes.

---

## 0 · The GitHub confusion, cleared up first

There are two different things called "connecting GitHub", and you only need one.

| | What it is | Do you need it? |
|---|---|---|
| **claude.ai → Settings → Connectors → GitHub** | Lets **claude.ai web sessions** read your repos | **No** |
| **Your machine's own git credentials** | What `git push` already uses | **Yes — you already have it** |

Claude Code CLI runs on your laptop and shells out to `git` using whatever
credentials your machine has. If `git push` works in your terminal, it works in
Claude Code. Nothing to connect.

Set up the GitHub connector only if you also want to use claude.ai in a browser
to look at the repo. It is not needed for any of the work below.

---

## 1 · Which surface

Claude Code runs in several places. For this project, use the **terminal CLI**.

| Surface | Verdict |
|---|---|
| **Terminal CLI** | **Use this.** Sees your files, runs your commands, uses your git. Everything below assumes it |
| VS Code / JetBrains extension | Fine if you live in an IDE — same engine, inline diffs |
| Desktop app | Good for reviewing diffs visually, multiple sessions side by side |
| Web (claude.ai/code) | Needs the GitHub connector; no access to your local frame cache |

All surfaces share the same `CLAUDE.md`, settings and MCP servers, so you can
mix them.

---

## 2 · Install

**macOS / Linux / WSL**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell** (prompt shows `PS C:\`)

```powershell
irm https://claude.ai/install.ps1 | iex
```

**Windows CMD** (prompt shows `C:\`, no `PS`)

```batch
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

Or via a package manager: `brew install --cask claude-code` ·
`winget install Anthropic.ClaudeCode` · apt/dnf/apk on Linux.

The install shows no progress while downloading. When it finishes, **open a new
terminal** and check:

```bash
claude --version
```

Prints a version followed by `(Claude Code)`. If the shell says `claude` is not
found, the install directory is not on your PATH yet — see
[Fix your PATH](https://code.claude.com/docs/en/troubleshoot-install).

Native installs auto-update in the background. Homebrew and WinGet do not —
`brew upgrade claude-code` / `winget upgrade Anthropic.ClaudeCode` occasionally.

> **Windows note:** install [Git for Windows](https://git-scm.com/downloads/win)
> so Claude Code gets a proper Bash tool. Without it, it falls back to
> PowerShell and some of the commands in `RUNBOOK.md` will need translating.

---

## 3 · Log in with the work account

```bash
cd badminton-agentic-vlm
claude
```

First run opens a browser to authenticate. Sign in with the **work account** —
that is the one with the higher limits.

Credentials are stored; you will not log in again. To switch accounts later,
type `/login` inside a running session.

Needs a Claude Pro, Max, Team or Enterprise subscription, or a Console account.

---

## 4 · Put `CLAUDE.md` in the repo root

This is the step that matters most.

```bash
cp ~/Downloads/CLAUDE.md ./CLAUDE.md
git add CLAUDE.md
git commit -m "docs: project context for Claude Code"
git push
```

Claude Code reads `CLAUDE.md` from the repo root at the start of **every**
session — yours and all three teammates'. It carries the twelve rules, the
dataset facts, the traps, and the current sprint. Nobody re-pastes context, and
nobody's session drifts from anybody else's.

Keep it current. When you pin the FBBench schema, when a new gotcha bites, when
the sprint rolls to Review 3 — update it and push.

---

## 5 · Verify the setup

```bash
claude
```

Then ask:

```
What is this project and what sprint are we in?
```

If it answers with the badminton project, Review 2 and the 21 Oct date, it read
`CLAUDE.md` and you are set. If it gives a generic answer, the file is not in
the repo root.

---

## 6 · Daily workflow

```bash
cd badminton-agentic-vlm
git checkout <your-branch>      # gouri / amal / vishal / sabeeh
git pull origin main            # start from current main
claude
```

Work. Then, at standup:

```bash
git checkout main && git pull
git merge <your-branch> && git push
```

Useful commands:

| Command | What |
|---|---|
| `claude` | Start a session |
| `claude "fix the scorer"` | Start with a prompt |
| `claude -c` | Continue the last conversation in this directory |
| `claude -r` | Resume a specific earlier conversation |
| `/clear` | Wipe context — use between unrelated tasks |
| `/help` | Commands available |
| `Shift+Tab` | Cycle permission mode (how much it does without asking) |

---

## 7 · Getting good results on this project

**Name the roadmap phase.** `CLAUDE.md` says the current sprint is Review 2.
Reinforce it:

> We're in Review 2 — zero-shot benchmark only, no agentic loop. I need
> `scripts/00_verify_dataset.py` run against the real data, and the overlap
> check interpreted.

**Make mock mode the gate, explicitly.** Early in a session:

> Before suggesting anything that runs on Kaggle, make mock mode pass:
> oracle 100%, random ~25%. If either is off, we fix the scorer first.

**Let it look before it builds.** For anything touching the real data:

> First run `describe_schema()` on `data/fbbench` and show me the actual field
> names. Don't change `fbbench.py` until we've both seen them.

**Break big tasks into numbered steps.** It handles sequences well:

> 1. extract frames for 10 clips only
> 2. show me what landed on disk
> 3. if it looks right, run the full extraction

**`/clear` between unrelated tasks.** Debugging the scorer and then building the
deck in one long session makes it slower and less accurate. Clear and restart.

**Correct it immediately.** If it drifts toward redesigning the project or
suggests building the agentic loop, say so plainly — `CLAUDE.md` says not to,
but a long session can wander.

---

## 8 · What not to use it for

| | Why |
|---|---|
| Running the GPU benchmark | Your laptop has no usable GPU; that lives on Kaggle |
| Clicking through the Kaggle UI | No browser access to your Kaggle account |
| Deciding the research direction | Settled, defended at a panel. Implementation only |
| Anything in `CLAUDE.md`'s trap table | Those are decided; re-litigating them wastes the sprint |

---

## 9 · Worth adding once things are running

Not now — in a quiet moment after Gate G2.

**Custom skills** for things you repeat. A `/verify` skill that runs the tests
plus mock mode and reports, or a `/newrun` skill that wires a result into the
manifest. See [skills](https://code.claude.com/docs/en/skills).

**A hook** that runs `pytest` before every commit, so `main` can't break. See
[hooks](https://code.claude.com/docs/en/hooks).

Both are ten-minute jobs that save more than they cost — but only after the
pipeline works. Don't build tooling before you have something to tool.

---

## First session, right now

```bash
cd badminton-agentic-vlm
cp ~/Downloads/CLAUDE.md .
git add CLAUDE.md && git commit -m "docs: project context" && git push
claude
```

Then paste the kickoff message from `KICKOFF.md`, attaching `HANDOFF.md` and
`RUNBOOK.md`.

First real task, and it is blocking everything else:

> Run `scripts/00_verify_dataset.py` against the downloaded data and tell me
> whether any clips appear in both Finebadminton-20K and FBBench.
