# Contributing to SageMaker ML Model Example

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/bojanderson/sagemaker-mlmodel-example.git
cd sagemaker-mlmodel-example
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-cdk.txt
```

3. Install development dependencies:
```bash
pip install pytest pytest-cov black flake8
```

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/unit/test_data_preparation.py -v
```

## Code Style

We use `black` for code formatting and `flake8` for linting:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/ --max-line-length=100
```

## Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and add tests

3. Run tests to ensure everything works:
```bash
pytest tests/ -v
```

4. Format your code:
```bash
black src/ tests/
```

5. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

6. Push and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Pull Request Guidelines

- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow existing code style
- Write clear commit messages

## Project Structure

```
.
├── src/
│   ├── pipeline/          # ML pipeline components
│   ├── cdk/              # Infrastructure code
│   └── run_pipeline.py   # Main orchestration script
├── scripts/              # Training scripts
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
└── data/                # Generated data (gitignored)
```

## Adding New Features

### Adding a New Model

1. Add model configuration in `src/pipeline/training.py`
2. Create training script if needed in `scripts/`
3. Update `run_pipeline.py` to include the new model
4. Add tests in `tests/unit/`

### Adding New Metrics

1. Update `src/pipeline/model_comparison.py`
2. Add logic to handle the new metric type
3. Add tests to verify metric comparison

## Questions?

Feel free to open an issue for questions or discussions about contributing!
