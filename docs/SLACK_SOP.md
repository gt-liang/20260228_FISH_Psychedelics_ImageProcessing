# Slack + GitHub Progress SOP

A lightweight system so labmates can track your work **without** you giving verbal updates.

---

## Setup (one-time)

1. In your Slack channel (e.g. `#proj-psychedelics`), run:
   ```
   /github subscribe boeynaems-lab/20260228_FISH_Psychedelics_ImageProcessing pulls
   ```
   This auto-posts to Slack whenever a PR is opened, updated, or merged.

2. Pin the repo link in the channel header.

---

## Workflow: One Task = One Thread + One PR

### Step 1 — Open a Draft PR immediately (even with zero code)
```bash
git checkout -b wip/module1-tif-reader
git commit --allow-empty -m "wip: start module1 TIF reader"
git push -u origin wip/module1-tif-reader
gh pr create --draft --title "WIP: Module 1 — TIF reader for Hyb rounds" \
  --body "$(cat <<'EOF'
## Goal
Adapt Module 1 preprocessing to read `.tif` input (Hyb2/3/4) instead of `.czi`.

## Done
- [ ] Nothing yet

## Next
- [ ] Implement TIF loader
- [ ] MIP projection
- [ ] QC checks

## Blockers
None

🤖 Tracked via Claude Code
EOF
)"
```

### Step 2 — Post the "task main thread" in Slack

Copy-paste this template as your **first message** in `#proj-psychedelics`:

```
🔬 [Task] Module 1 — TIF reader for Psychedelics FISH
PR: <link>

Done: —
Next: Implement TIF loader + MIP
Blockers: None

Will update here as work progresses 👇
```

### Step 3 — Daily update in the same thread (3 lines, 30 sec)

```
Done: TIF loader working for Hyb2; MIP verified visually
Next: Extend to Hyb3/4, add QC metadata
Blockers: None
```

### Step 4 — When PR is ready for review

Change Draft → Ready for Review on GitHub. Slack auto-posts the event.
Add a final thread message:
```
✅ Ready for review — @labmate can you take a look when you have a chance?
```

---

## PR Description Template

Keep this at the top of every PR description:

```markdown
## Done
- ...

## Next
- ...

## Blockers
- None / ...

## Scientific Logic
_Why does this code do what it does biologically?_

## Test / Sanity Check
_How was this verified? Expected output?_
```

---

## Branch Naming

| Type | Format | Example |
|------|--------|---------|
| Work in progress | `wip/<topic>` | `wip/module1-tif-reader` |
| New feature | `feat/<topic>` | `feat/puncta-detection` |
| Bug fix | `fix/<topic>` | `fix/registration-shift-sign` |

---

## What NOT to post in Slack/GitHub

- No raw data paths (OneDrive paths)
- No tokens or credentials
- No patient/sample identifiers beyond coded names (e.g., `psy_redo3`)
