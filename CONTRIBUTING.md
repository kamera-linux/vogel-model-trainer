# Contributing to Vogel Model Trainer

Thank you for your interest in contributing to vogel-model-trainer! 🎉

We welcome contributions from the community and appreciate your effort to make this project better.

## 🌟 Ways to Contribute

- 🐛 **Report Bugs**: Submit detailed bug reports via [GitHub Issues](https://github.com/kamera-linux/vogel-model-trainer/issues)
- 💡 **Suggest Features**: Share ideas for new features or improvements
- 📝 **Improve Documentation**: Help make our docs clearer and more comprehensive
- 🔧 **Submit Code**: Fix bugs or implement new features
- 🧪 **Write Tests**: Improve test coverage
- 🌍 **Translations**: Help translate documentation (future)

## 📋 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/vogel-model-trainer.git
cd vogel-model-trainer

# Add upstream remote
git remote add upstream https://github.com/kamera-linux/vogel-model-trainer.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
vogel-trainer --version
```

### 3. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

## 💻 Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused
- Use type hints where appropriate

Example:
```python
def extract_birds(
    video_path: str,
    output_dir: str,
    bird_species: str,
    confidence_threshold: float = 0.3
) -> int:
    """
    Extract bird images from a video for training dataset creation.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted bird images
        bird_species: Name of the bird species (for organizing)
        confidence_threshold: Minimum YOLO detection confidence (default: 0.3)
        
    Returns:
        Number of bird images extracted
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If confidence_threshold is not between 0 and 1
    """
    # Implementation here
    pass
```

### Testing

- Write tests for new features
- Ensure all tests pass before submitting
- Aim for good test coverage

```bash
# Run tests (when test suite is available)
pytest

# Run tests with coverage
pytest --cov=vogel_model_trainer
```

### Documentation

- Update README.md if you change user-facing features
- Update docstrings for code changes
- Add comments for complex logic
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated (README, docstrings)
- [ ] Tests added/updated and passing
- [ ] CHANGELOG.md updated
- [ ] Commits are clean and well-described

### Commit Messages

Write clear commit messages:

```bash
# Good commit messages
git commit -m "Add auto-sorting mode for training data extraction"
git commit -m "Fix memory leak in video processing"
git commit -m "Update README with new CLI examples"

# Bad commit messages (avoid these)
git commit -m "fix bug"
git commit -m "changes"
git commit -m "wip"
```

### Submitting

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template:
   - Describe your changes
   - Reference related issues
   - Explain testing performed
   - Add screenshots if applicable

4. Wait for review and address feedback

### After Submission

- Respond to review comments
- Make requested changes
- Keep your branch updated with main:
  ```bash
  git fetch upstream
  git rebase upstream/main
  git push --force-with-lease
  ```

## 🐛 Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: Detailed steps
- **Expected vs Actual Behavior**: What should happen vs what does happen
- **Environment**: OS, Python version, package version
- **Training/Dataset Information**: Model, dataset size, number of classes
- **Error Output**: Full error messages and tracebacks
- **Additional Context**: Logs, screenshots, etc.

Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md)

## 💡 Suggesting Features

When suggesting features, please include:

- **Problem/Use Case**: What problem does it solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions considered
- **Benefits**: Who would benefit and how?
- **Examples**: Code examples or mockups

Use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md)

## 🔒 Security Issues

**Do not open public issues for security vulnerabilities.**

Please report security issues privately via:
- Email: kamera-linux@mailbox.org
- GitHub Security Advisories

See our [Security Policy](SECURITY.md) for details.

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment, discriminatory language, or personal attacks
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

### Enforcement

Project maintainers reserve the right to:
- Remove, edit, or reject contributions that violate this code
- Ban contributors for inappropriate behavior

## ❓ Questions?

- Check the [README](README.md) and [documentation](https://github.com/kamera-linux/vogel-model-trainer)
- Search [existing issues](https://github.com/kamera-linux/vogel-model-trainer/issues)
- Ask questions in [GitHub Discussions](https://github.com/kamera-linux/vogel-model-trainer/discussions)
- Open a [Question issue](.github/ISSUE_TEMPLATE/custom.md)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make this project better for everyone. Thank you for taking the time to contribute!

---

**Happy Coding! 🐦💻**
