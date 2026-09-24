#!/usr/bin/env bash
# ViMedAgentv2 — block every git command that WRITES. Inspection only.
#
# Why: this directory holds experiment results that cannot be cheaply
# regenerated (results/**/*.jsonl, ~10 MB per seed, tens of GPU-hours per
# 3-seed set; mdagents_vi x sglang-qwen3-32b alone took ~99,000 s).
# One stray `git add . && git commit && git push` publishes the whole
# corpus; one stray `git reset --hard` or `git clean -fd` destroys it,
# and there is no second copy.
#
# Contract: read the PreToolUse hook JSON on stdin, print a decision JSON
# on stdout. Silence (exit 0, no output) means "no opinion" -> allowed.
set -uo pipefail

cmd="$(jq -r '.tool_input.command // empty' 2>/dev/null)"
[ -z "$cmd" ] && exit 0

deny() {
  jq -n --arg r "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }'
  exit 0
}

# Allow -C <path>, -c <k=v>, --git-dir=... between "git" and its subcommand.
OPT='(\s+(-[A-Za-z]+(\s+[^ ]+)?|--[^ ]+))*'
# Catch the command at line start or hidden behind ; && || | ` $( .
PRE='(^|[;&|`]|\$\(|\s)'

# 1) Unambiguously writing subcommands.
WRITE='add|commit|push|init|clone|am|apply|cherry-pick|rebase|reset|revert|merge|stash|rm|mv|restore|checkout|switch|clean|gc|prune|fetch|pull|submodule|worktree|update-ref|filter-branch|filter-repo|update-index|write-tree|commit-tree|hash-object|fast-import'
if printf '%s' "$cmd" | grep -Eq "${PRE}git${OPT}\s+(${WRITE})(\s|$)"; then
  verb="$(printf '%s' "$cmd" | grep -Eo "git${OPT}\s+(${WRITE})(\s|$)" | grep -Eo "(${WRITE})" | head -1)"
  deny "BLOCKED in ViMedAgentv2: 'git ${verb}'. This directory is inspect-only; writing to git is not permitted (see CLAUDE.md). Read-only commands still work: git status / diff / log / show / ls-files. If a write is genuinely required, the user must run it by hand."
fi

# 2) Subcommands that both read and write — block only the writing forms.
if printf '%s' "$cmd" | grep -Eq "${PRE}git${OPT}\s+(branch|tag)\s+.*(-d|-D|-m|-M|-f|--delete|--force|--move)(\s|$)"; then
  deny "BLOCKED in ViMedAgentv2: 'git branch/tag' in delete/rename/force form. Listing is allowed (git branch, git tag). See CLAUDE.md."
fi
if printf '%s' "$cmd" | grep -Eq "${PRE}git${OPT}\s+remote\s+(add|remove|rm|set-url|rename|prune)(\s|$)"; then
  deny "BLOCKED in ViMedAgentv2: modifying git remotes. Viewing is allowed (git remote -v). See CLAUDE.md."
fi
if printf '%s' "$cmd" | grep -Eq "${PRE}git${OPT}\s+config\s+" \
   && ! printf '%s' "$cmd" | grep -Eq "git${OPT}\s+config\s+(--get|--list|-l|--get-all|--get-regexp)(\s|$)"; then
  deny "BLOCKED in ViMedAgentv2: 'git config' in writing form. Reading is allowed (git config --get / --list). See CLAUDE.md."
fi

exit 0
