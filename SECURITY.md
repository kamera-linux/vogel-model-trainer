# Security Policy

## 🔐 Supported Versions

We actively support the following versions of vogel-model-trainer:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

If you discover a security vulnerability, please report it privately via:

1. **Email**: kamera-linux@mailbox.org
2. **GitHub Security Advisories**: [Report a vulnerability](https://github.com/kamera-linux/vogel-model-trainer/security/advisories/new)

### What to Include

Please include as much information as possible:

- **Type of vulnerability** (e.g., code injection, model poisoning, data leakage)
- **Affected component** (e.g., training module, extraction, CLI)
- **Affected versions**
- **Steps to reproduce** the vulnerability
- **Potential impact** of the vulnerability
- **Suggested fix** (if you have one)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Target**: Depends on severity
  - Critical: ASAP (within days)
  - High: Within 2 weeks
  - Medium: Within 30 days
  - Low: Next release

### Disclosure Policy

- **Embargo Period**: We request 90 days before public disclosure
- **Coordinated Disclosure**: We'll work with you on timing
- **Credit**: We'll acknowledge your contribution (unless you prefer anonymity)

## 🛡️ Security Best Practices for Users

### When Using vogel-model-trainer:

1. **Keep Updated**: Always use the latest version
   ```bash
   pip install --upgrade vogel-model-trainer
   ```

2. **Validate Input**: Be cautious with video files and datasets from untrusted sources
   - Malicious video files could exploit vulnerabilities in OpenCV or other dependencies
   - Poisoned datasets could compromise model training

3. **Model Files**: Only use YOLO and trained model files from trusted sources
   - Verify model checksums when available
   - Be cautious loading models from unknown sources

4. **File Operations**: Be aware of file operations
   - Extraction creates many image files - ensure sufficient disk space
   - Training can create large checkpoint files
   - Run with appropriate user permissions

5. **Dependency Security**: Keep dependencies updated
   ```bash
   pip install --upgrade opencv-python ultralytics torch torchvision
   ```

6. **Training Data Privacy**: Be mindful of privacy
   - Training videos may contain sensitive information
   - Consider data retention policies
   - Don't share private datasets publicly

7. **Virtual Environments**: Use isolated environments
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install vogel-model-trainer
   ```

## 🔍 Known Security Considerations

### Model Training

- **Resource Exhaustion**: Large datasets can consume significant memory and disk
- **Data Poisoning**: Malicious training data can compromise model behavior
- **Model Theft**: Trained models may contain intellectual property

### Video Processing

- **Codec Vulnerabilities**: Video codecs may have security vulnerabilities
- **Memory Issues**: Large videos can cause memory exhaustion
- **Path Traversal**: Always validate file paths

### Dependencies

We rely on several third-party libraries:
- **PyTorch**: Deep learning framework
- **OpenCV**: Video processing (may have codec vulnerabilities)
- **Ultralytics**: YOLOv8 implementation
- **Transformers**: Model training

These dependencies may have their own security considerations.

## 🔧 Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1 → 0.1.2)
- Announced in CHANGELOG.md
- Tagged with `security` label in GitHub releases
- Communicated via GitHub Security Advisories

## 📋 Security Checklist for Contributors

When contributing code, please:

- [ ] Validate all user inputs
- [ ] Sanitize file paths
- [ ] Handle errors securely (don't expose sensitive info)
- [ ] Avoid hardcoded credentials or secrets
- [ ] Use secure defaults
- [ ] Document security implications
- [ ] Run security linters (e.g., `bandit`)

## 🤝 Responsible Disclosure

We appreciate responsible disclosure and will:

- Acknowledge your report within 48 hours
- Keep you informed of our progress
- Credit you in the security advisory (if desired)
- Work with you on coordinated disclosure timing

## 📚 Additional Resources

- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories)

## 📞 Contact

For security-related questions or concerns:

- **Email**: kamera-linux@mailbox.org
- **Security Advisories**: [GitHub](https://github.com/kamera-linux/vogel-model-trainer/security/advisories)

---

**Thank you for helping keep vogel-model-trainer and its users safe!** 🙏
