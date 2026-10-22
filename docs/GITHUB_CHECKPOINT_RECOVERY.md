 # GitHub checkpoint recovery for LawLM QA V2

LawLM can keep rolling recovery checkpoints in a GitHub Release so a new machine can download the latest saved state and resume training. Checkpoints are release assets; they are not committed to the Git source tree.

## 1. Create a GitHub token

Create a fine-grained personal access token for the `ishitaarya/LawLM-` repository with **Contents: Read and write** permission. The token is used to create a release, upload assets, list assets, download assets, and delete old assets. Treat it like a password.

Never paste the token into Python source code, a committed `.env` file, a notebook, or a chat.

## 2. Set the token in PowerShell

In the same PowerShell window used to run training:

```powershell
$env:LAWLM_GITHUB_TOKEN = "PASTE_YOUR_TOKEN_HERE"
git pull origin main
python -m src.training.train_qa_v2
```

The environment variable applies to that PowerShell window. If you open a new window, set it again before starting training.

Optional overrides: `LAWLM_GITHUB_REPO` defaults to `ishitaarya/LawLM-`; `LAWLM_GITHUB_RELEASE_TAG` defaults to `qa-v2-checkpoints`.

## 3. What gets backed up

- Local recovery checkpoints are saved every 50 optimizer updates.
- A GitHub Release backup is uploaded every 250 optimizer updates and at the end of an epoch.
- The newest five remote checkpoint assets are retained.
- Each recovery checkpoint contains model weights, optimizer state, epoch and batch position, sample order, and RNG state.
- At startup, training checks GitHub and downloads a newer remote checkpoint if one exists. It can resume from that checkpoint without loading the original base-model checkpoint.

The regular best-validation checkpoint remains a separate local file: `checkpoints/lawsuit_llm_qa_v2_best.pt`.

## 4. Recovery limitations

GitHub backups only exist after an upload completes. If the process or network fails before the next remote upload, recovery resumes from the most recent remote checkpoint and repeats the unsaved work. Keep the local `checkpoints/qa_v2_resume/` directory too.

The training data and tokenizer must also be available on the recovery machine. If `LAWLM_GITHUB_TOKEN` is missing or invalid, training continues with local checkpoints and prints a warning; no remote backup is performed.
