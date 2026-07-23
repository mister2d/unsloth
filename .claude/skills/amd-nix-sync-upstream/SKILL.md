---
name: amd-nix-sync-upstream
description: Run the full AMD/ROCm devenv upstream-sync pipeline — merge latest unslothai/main, dispatch amd-nix-maintainer to fix/regenerate the lock+SBOM, amd-nix-sbom-auditor to risk-assess the change, and amd-nix-validator to confirm on real hardware. Use when asked to sync the nix/ AMD devenv with upstream, or after merging new commits that touch files nix/ depends on (studio/backend/requirements/*.txt, pyproject.toml, studio/backend/core/training/worker.py, hardware detection code).
---

Follow `nix/CLAUDE.md`'s "When this breaks after an upstream merge" procedure exactly, in order:

1. Confirm the current branch/worktree is up to date with the latest relevant upstream commits (merge if not already done — ask the user first if there's any ambiguity about which branch to merge from, per this repo's normal git safety practice).
2. Dispatch `amd-nix-maintainer` to check for and fix any breakage, regenerating `nix/requirements.lock.txt` and both SBOM files via `lock-deps` as part of its normal duties.
3. Dispatch `amd-nix-sbom-auditor` to diff the regenerated SBOMs against the prior committed versions and produce a risk-flagged report.
4. Dispatch `amd-nix-validator` for real-hardware confirmation, scoped by the auditor's report (light pass if low-risk-only, full checklist if anything is flagged high-risk).
5. Summarize the outcome for the user: what changed, what broke (if anything) and how it was fixed, the auditor's risk report, and the validator's go/no-go — then ask whether to commit and push, same as every prior round in this project. Never commit/push without that explicit confirmation, regardless of how clean the pipeline result is — this skill automates the *pipeline*, not the decision to ship it.
6. If any step reports a NO-GO that isn't resolved within the pipeline (e.g. a genuinely new failure mode nothing here anticipated), stop and hand back to the user with a clear summary rather than looping indefinitely — this mirrors how every real round in this project's actual history got resolved (iterative fix-and-revalidate with a human kept informed throughout), not a fully unattended retry loop.
