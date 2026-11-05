# Preserved local string-action version

These files are byte-identical copies of the local implementation, validator,
tests and documentation before the ACTION_ID change (Git base commit 63b5891).
They are historical source snapshots, not standalone entry points: their relative
imports and project paths assume the original repository layout. Use that Git
revision in a separate local checkout if reproducing the old implementation.

The active local agent now emits PLAN/ACTION_ID and maps the integer through
the current, unmodified admissible_actions list. Existing PDDL/OpenAI/Gemini
source and historical trajectories/logs remain unchanged.
