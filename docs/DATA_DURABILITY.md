# Data durability — never lose a file again

Large data files (parquets) are **not** stored in Git. They live in **GitHub Releases**
(immutable, off your laptop). `config/data_registry.json` records where each one lives
and its sha256, so any clone can pull and verify it byte-for-byte.

## The only two commands you need

```bash
# Get every dataset into a fresh/stale clone (verifies checksums):
python3 scripts/fetch_data.py --all
# or specific ones:
python3 scripts/fetch_data.py wnba_tracking wnba_hustle

# After you (re)generate a dataset, push it and update the registry:
python3 scripts/publish_data.py --name wnba_games   # then commit config/data_registry.json
```

`git pull` handles code. `fetch_data.py` handles data. Together, a clone is never a dead end.

## 3-2-1 backup (do once, then forget)

1. **Time Machine** to an external drive (local, automatic).
2. **Backblaze Personal** ($9/mo, unlimited) or iCloud/Dropbox on `data/` (offsite).
3. Datasets published to Releases via `publish_data.py` (versioned, off-machine).

## Rules

- Work from **one** canonical clone. Don't fan out across copies that drift.
- A working copy (laptop, cloud VM, server) is **disposable**. The Release + registry is the truth.
- `scripts/verify_data_registry.py` fails if a present file's hash drifts from the registry.
