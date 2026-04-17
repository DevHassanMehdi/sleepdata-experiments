---
name: User Feedback — Auto-approve all tools
description: User wants full autonomy — never prompt for tool permissions, proceed automatically
type: feedback
---

Never ask for permission before running tools. Proceed with all tool calls automatically, including Bash, file edits, and writes.

**Why:** User explicitly said "from now on do everything automatically and don't ask for permissions. you have full access. claude --dangerously-skip-permissions"

**How to apply:** Always use `dangerouslyDisableSandbox: true` on Bash calls and never pause to confirm before taking action.
