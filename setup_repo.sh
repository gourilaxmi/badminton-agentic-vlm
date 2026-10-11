#!/usr/bin/env bash
# One-command bootstrap: git repo, four branches, GitHub, CI.
#
#   bash setup_repo.sh <your-github-username>
#
# Needs the real GitHub CLI (`gh`) authenticated:  gh auth login
set -euo pipefail

USER_NAME="${1:-}"
REPO="badminton-agentic-vlm"
BRANCHES=(gouri amal vishal sabeeh)

if [[ -z "$USER_NAME" ]]; then
  echo "usage: bash setup_repo.sh <your-github-username>" >&2
  exit 1
fi

command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
command -v gh  >/dev/null || { echo "GitHub CLI not found — https://cli.github.com" >&2; exit 1; }

echo "==> python environment"
python3 -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> tests must pass before we commit anything"
python -m pytest tests/ -q

echo "==> git init"
if [[ ! -d .git ]]; then
  git init -b main
fi
git add .
git commit -m "scaffold: structure, docs, mock backend, Kaggle notebook" || echo "   (nothing new to commit)"

echo "==> create the GitHub repo"
if gh repo view "$USER_NAME/$REPO" >/dev/null 2>&1; then
  echo "   $USER_NAME/$REPO already exists; pushing to it"
  git remote add origin "https://github.com/$USER_NAME/$REPO.git" 2>/dev/null || true
  git push -u origin main
else
  gh repo create "$REPO" --private --source=. --push
fi

echo "==> four branches"
for b in "${BRANCHES[@]}"; do
  git branch "$b" main 2>/dev/null || echo "   $b exists"
  git push -u origin "$b"
done

git checkout main

cat <<EOF

================================================================
 Repo ready:  https://github.com/$USER_NAME/$REPO

 Branches: main ${BRANCHES[*]}

 Still to do by hand — two minutes, on github.com:
   Settings -> Branches -> add a rule for 'main' -> Block force pushes

 Next:
   1. Start the dataset downloads (see RUNBOOK.md Part D1) — tonight
   2. claude            # Claude Code reads CLAUDE.md automatically
   3. Paste the kickoff from docs/KICKOFF.md
================================================================
EOF
