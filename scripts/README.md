# Scripts

Automation scripts for vogel-model-trainer development and release management.

## 📦 create_github_release.py

Automated GitHub release creation from Git tags with PyPI workflow monitoring.

### Features

- 🏷️ Auto-detects latest Git tag or uses specified tag
- 📝 Finds and uses release notes from `.github/release-v*.md`
- 🚀 Creates GitHub release via `gh` CLI
- 📊 Monitors PyPI publish workflow status in real-time
- 🎯 Matches workflow by release tag/version (prevents showing wrong version)
- ⚡ Supports draft and pre-release options
- 🔄 Handles existing releases (delete/recreate)
- ✅ Validates tag format and remote sync
- 🔍 Searches last 5 workflow runs to find the correct one

### Usage

```bash
# Use latest tag
python scripts/create_github_release.py

# Specific tag
python scripts/create_github_release.py v0.1.15

# Create as draft
python scripts/create_github_release.py --draft

# Mark as pre-release
python scripts/create_github_release.py v0.2.0-beta.1 --prerelease

# Custom title
python scripts/create_github_release.py --title "Release v0.1.15: Enhanced Training"

# Skip confirmations
python scripts/create_github_release.py --force
```

### Requirements

- GitHub CLI (`gh`) installed and authenticated
- Git repository with tags
- Release notes in `.github/release-v*.md` (optional)

### Options

```
positional arguments:
  tag                   Git tag to release (default: latest tag)

optional arguments:
  --draft              Create release as draft
  --prerelease         Mark release as pre-release
  --title TITLE        Custom release title
  --notes-dir DIR      Directory for release notes (default: .github)
  --force              Skip all confirmation prompts
```

### Workflow

1. Detects/validates Git tag (e.g., `v0.1.15`)
2. Checks if tag exists locally and on remote
3. Searches for release notes: `.github/release-v0.1.15.md`
4. Extracts title from notes (H1 heading) or generates default
5. Creates GitHub release with notes
6. Waits for PyPI publish workflow to start (max 20s)
7. Finds workflow run matching the release tag/version
8. Displays workflow status with real-time updates

### Release Notes File Patterns

The script searches for release notes files in multiple patterns:
- `.github/release-v0.1.15.md` (preferred)
- `.github/release-0.1.15.md`
- `.github/RELEASE-v0.1.15.md`
- `.github/RELEASE-0.1.15.md`

### Example Output

```
🎯 GitHub Release Creator
============================================================

🔍 Searching for latest Git tag...
✅ Found latest tag: v0.1.15
✅ Tag exists on remote
📝 Looking for release notes file...
✅ Found release notes: .github/release-v0.1.15.md
📋 Extracted title: v0.1.15 - Enhanced Training & PNG Support

📊 Release Summary:
============================================================
Tag:         v0.1.15
Title:       v0.1.15 - Enhanced Training & PNG Support
Notes:       .github/release-v0.1.15.md
Draft:       False
Pre-release: False
============================================================

Create this release? [y/N]: y

🚀 Creating GitHub release for v0.1.15...
✅ Release created: https://github.com/kamera-linux/vogel-model-trainer/releases/tag/v0.1.15

============================================================
Checking for PyPI publish workflow...
============================================================
⏳ Waiting for 'publish-pypi.yml' workflow to start for v0.1.15...
✅ Workflow found for v0.1.15 (Run ID: 19444578265)

============================================================
📊 PyPI Publish Workflow Status
============================================================
Workflow:   Publish to PyPI
Title:      v0.1.15 - Enhanced Training & PNG Support
Status:     ✅ completed
Result:     ✅ SUCCESS
URL:        https://github.com/kamera-linux/vogel-model-trainer/actions/runs/19444578265
============================================================

💡 Useful commands:
   Watch live:  gh run watch 19444578265
   View logs:   gh run view 19444578265 --log
   Open web:    gh run view 19444578265 --web
```

### Notes

- Script automatically pushes unpushed tags if confirmed
- Validates semantic versioning format (`v0.1.15` or `0.1.15`)
- Can delete and recreate existing releases
- Monitors GitHub Actions workflow for PyPI publishing
- Matches workflow by tag/version (searches last 5 runs)
- Uses auto-generated notes if release notes file not found
- Draft releases do not trigger PyPI workflow monitoring
- Workflow status shows: queued ⏳, in_progress 🔄, completed ✅
- Result icons: success ✅, failure ❌, cancelled 🚫, skipped ⏭️

## 📚 Related Documentation

- **GitHub CLI Guide**: See `.github/GITHUB_CLI_GUIDE.md` for comprehensive `gh` CLI documentation
- **Release Workflow**: Complete guide for releases, PRs, workflows, and automation
- **Best Practices**: Authentication, versioning, and scripting guidelines
