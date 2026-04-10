# Contributing to QCS

Thank you for your interest in contributing to the Quantitative Commodity Trading System.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check the [existing issues](../../issues) to see if the bug has been reported
2. Update your packages to the latest versions
3. Include your environment details (Python version, OS, package versions)

When submitting a bug report, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Log files or error messages
- Environment information

### Suggesting Enhancements

We welcome suggestions for new features. Please submit a detailed proposal including:
- Use case description
- Proposed implementation approach
- Potential impact on existing functionality

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions
- Include type hints where appropriate

### Testing

Before submitting a PR:
- Run existing tests to ensure no regressions
- Add tests for new functionality
- Ensure all tests pass

```bash
pytest tests/
```

## Project Structure

Please follow the established project structure:
- `src/` - Source code (environments, models, strategies, utils)
- `scripts/` - Executable scripts
- `data/` - Data files
- `results/` - Experimental results
- `docs/` - Documentation
- `tests/` - Unit tests

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
