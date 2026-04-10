# r60 Preservation Note

This folder preserves the authoritative output artifacts for the historical `r60`
classical-CV checkpoint.

## What `r60` means

`r60` is the best known CV-only checkpoint found during iterative development on
2026-04-09 before later experiments (`r62` onward) failed to improve it
consistently. It is **not** the overall project accuracy baseline; the accepted
accuracy baseline remains `v8` from the Cellpose path. `r60` is the CV baseline
that all later CV experiments should compare against.

## Authoritative local artifact root

The authoritative `r60` outputs live outside the repo at:

```text
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1
```

Important files under that directory:

```text
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1/counts.csv
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1/candidate_table.csv
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1/review_queue.csv
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1/reports/run_summary.md
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1/overlays
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60/baseline_v1/intermediate
```

This repo folder preserves two copies from that run:

- `counts_r60.csv`
- `run_summary_r60.md`

## Why this preservation note exists

During later work, the exact code/config state that produced `r60` was not fully
recovered from the working tree. The output artifacts above are therefore the
authoritative record.

Two explicit reconstruction attempts were made and both failed to match the
historical `r60` counts:

1. Current CV code with later heuristics removed:

```text
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_r60_preserve_check/baseline_v1
```

2. Older handoff snapshot config applied to current code:

```text
/Users/stephenyu/Documents/New project/data/processed/cv_backend_compare_2026-04-09_snapshot_preserve_probe/baseline_v1
```

Those runs should be treated only as failed reconstruction probes, not as `r60`.

## Focus-image counts from the authoritative `r60`

These five images were repeatedly used for manual inspection:

| image_id | total | middle | top_5pct |
| --- | ---: | ---: | ---: |
| `scan_20260313_10_4ffe35b016` | 79 | 48 | 1 |
| `scan_20260313_22_74a4d07f84` | 123 | 108 | 1 |
| `scan_20260313_25_db0b949a97` | 115 | 85 | 2 |
| `scan_20260313_7_78157780df` | 104 | 71 | 2 |
| `scan_20260313_74_f670a25628` | 102 | 78 | 6 |

## How later CV work should use `r60`

When evaluating any future CV iteration:

1. Compare it directly against the accepted Cellpose baseline `v8`.
2. Compare it directly against the preserved `r60` outputs.
3. Do not claim an improvement based only on higher totals.
4. Always inspect original image, accepted overlay, and failure regions at high
   zoom, especially on:
   - `10`: ordinary-area pair merge
   - `25`: dense undercount
   - `7`: broad recall failure
   - `74`: mixed merge / local dense issues

## Manual inspection method that was used

The development process repeatedly relied on:

1. Opening original images and overlays side by side.
2. Zooming into suspicious local regions instead of trusting image-level totals.
3. Checking whether a count increase came from:
   - real missed pupa recovery
   - edge junk
   - blue-dot confusion
   - splitting one pupa into two
4. Rejecting any iteration that improved one image by globally over-splitting the
   rest.

## Current repo context at preservation time

Repo root:

```text
/Users/stephenyu/Documents/pupa_counter_publish
```

This preservation branch exists to keep the `r60` record safe before further CV
optimization continues elsewhere.
