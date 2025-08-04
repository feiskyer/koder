# Contributing to Koder

Welcome to Koder! We're excited that you want to contribute to this experimental AI coding assistant. This project is a learning-focused exploration of building advanced terminal-based AI agents, and we welcome contributions of all kinds.

## Project Philosophy

Koder is designed as both a functional tool and a learning playground for AI agent development. We value:

- **Experimentation**: Trying new approaches and learning from them
- **Clean Architecture**: Well-structured, maintainable Python code
- **Security First**: Robust validation and permission systems
- **Universal Compatibility**: Supporting multiple AI providers and use cases
- **Community Learning**: Sharing knowledge and growing together

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Git for version control
- API key for at least one AI provider (OpenAI, Anthropic, Google, etc.)

### Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/feiskyer/koder.git
   cd koder
   ```

2. **Set Up Environment**

   ```bash
   # Install dependencies
   uv sync

   # Configure AI provider (example: OpenAI)
   export OPENAI_API_KEY="your-api-key"
   export KODER_MODEL="gpt-4o"
   ```

3. **Verify Installation**

   ```bash
   # Test the CLI
   uv run koder "Hello, Koder!"

   # Run in interactive mode
   uv run koder
   ```

4. **Run Development Commands**

   ```bash
   # Code formatting
   black .

   # Linting and fixes
   ruff check --fix

   # Error-only pylint check
   pylint koder_agent/ --disable=C,R,W --errors-only
   ```

## Development Guidelines

### Code Style

We use automated tooling to maintain consistent code quality:

- **Black**: Code formatting (`black .`)
- **Ruff**: Linting and import sorting (`ruff check --fix`)
- **Pylint**: Additional error checking (`pylint koder_agent/ --disable=C,R,W --errors-only`)

### Code Quality Standards

1. **Type Hints**: Use type annotations for function parameters and return values
2. **Docstrings**: Document all public functions and classes
3. **Error Handling**: Implement proper exception handling with informative messages
4. **Security**: Follow SecurityGuard patterns for input validation
5. **Testing**: Add tests for new features (when applicable)

### Architecture Principles

1. **Separation of Concerns**: Keep tools, core logic, and UI separate
2. **Plugin Architecture**: New tools should extend the tool engine
3. **Security by Default**: All user inputs must be validated
4. **Provider Agnostic**: Support multiple AI providers through abstraction
5. **Rich UI**: Use Rich library for terminal interfaces

## Making Contributions

1. **Create an Issue** (for significant changes)

   - Describe the problem or feature request
   - Discuss the approach with maintainers
   - Get feedback before implementation

2. **Fork and Branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Develop and Test**

   - Write clean, documented code
   - Run code quality checks
   - Test your changes thoroughly
   - Update documentation if needed

4. **Commit and Push**

   ```bash
   # Make atomic, descriptive commits
   git add .
   git commit -m "feat: add new tool for X functionality"

   # Push to your fork
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what you changed and why
   - Include testing information

## Security Considerations

### Security Requirements

1. **Input Validation**: All user inputs must be validated
2. **Command Safety**: Shell commands require SecurityGuard approval
3. **API Key Protection**: Never log or expose API keys
4. **Output Filtering**: Filter sensitive information from outputs
5. **Permission Checks**: Tool execution must check permissions

### Reporting Security Issues

For security vulnerabilities:

1. **DO NOT** create public issues
2. Email maintainers privately
3. Provide detailed reproduction steps
4. Allow time for responsible disclosure

## Community Guidelines

### Code of Conduct

- **Be Respectful**: Treat all community members with respect
- **Be Inclusive**: Welcome contributors of all backgrounds and skill levels
- **Be Collaborative**: Work together and help each other learn
- **Be Patient**: Remember that everyone is learning

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Reviews**: Ask for feedback on your contributions
