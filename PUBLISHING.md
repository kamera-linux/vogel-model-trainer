# Publishing to PyPI

This document describes how to publish `vogel-model-trainer` to PyPI.

## Prerequisites

### 1. PyPI Account Setup

1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [Test PyPI](https://test.pypi.org/account/register/) (testing)

2. Enable 2FA on both accounts

3. Generate API tokens:
   - PyPI: Account Settings → API tokens → "Add API token"
   - Test PyPI: Account Settings → API tokens → "Add API token"

### 2. Configure Credentials

#### For Local Publishing

Copy the example config:
```bash
cp .pypirc.example ~/.pypirc
chmod 600 ~/.pypirc
```

Edit `~/.pypirc` and add your API tokens:
```ini
[pypi]
username = __token__
password = pypi-YOUR-ACTUAL-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

#### For GitHub Actions

Add secrets to your GitHub repository:
1. Go to Settings → Secrets and variables → Actions
2. Add these secrets:
   - `PYPI_API_TOKEN`: Your PyPI API token
   - `TEST_PYPI_API_TOKEN`: Your Test PyPI API token

## Version Management

Update the version in `src/vogel_model_trainer/__version__.py`:
```python
__version__ = "0.2.0"  # Update this
```

Also update `pyproject.toml`:
```toml
[project]
version = "0.2.0"  # Keep in sync with __version__.py
```

Update `CHANGELOG.md` with release notes.

## Publishing Process

### Option 1: Manual Publishing (Local)

#### Step 1: Test on Test PyPI

```bash
# Build the package
./scripts/build.sh

# Upload to Test PyPI
./scripts/upload-testpypi.sh

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            vogel-model-trainer
```

#### Step 2: Publish to PyPI

```bash
# Upload to PyPI
./scripts/upload-pypi.sh
```

### Option 2: Automated Publishing (GitHub Actions)

#### For Testing (Test PyPI)

1. Go to Actions → "Publish to PyPI"
2. Click "Run workflow"
3. This will publish to Test PyPI

#### For Production Release (PyPI)

1. Create and push a git tag:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. Create a GitHub Release:
   - Go to Releases → "Draft a new release"
   - Choose your tag (v0.2.0)
   - Write release notes
   - Click "Publish release"

3. GitHub Actions will automatically:
   - Build the package
   - Run checks
   - Upload to PyPI
   - Attach wheel and source files to the release

## Verification

After publishing, verify the package:

```bash
# Check PyPI page
open https://pypi.org/project/vogel-model-trainer/

# Test installation in a fresh environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
pip install vogel-model-trainer

# Test the command
vogel-trainer --version
```

## Troubleshooting

### Build fails

```bash
# Clean everything and rebuild
rm -rf build/ dist/ *.egg-info src/*.egg-info
./scripts/build.sh
```

### Version already exists on PyPI

You cannot overwrite a version on PyPI. Increment the version number and try again.

### Import errors after installation

Make sure your `pyproject.toml` has the correct package path:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/vogel_model_trainer"]
```

### Missing dependencies

Check that all dependencies are listed in `pyproject.toml` under `dependencies`.

## Checklist

Before publishing a new version:

- [ ] All tests pass
- [ ] Version bumped in `__version__.py` and `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] Documentation updated
- [ ] Tested on Test PyPI
- [ ] README includes correct installation instructions
- [ ] All files included in `MANIFEST.in`

## Resources

- [PyPI](https://pypi.org/)
- [Test PyPI](https://test.pypi.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
