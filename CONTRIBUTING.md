# Contributing to Feather DB

Thank you for your interest in contributing to Feather DB! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Environment details**:
  - Operating system and version
  - Python version (if using Python API)
  - Rust version (if using CLI)
  - Compiler version (if building from source)
- **Code sample** that demonstrates the issue (if applicable)
- **Error messages** or stack traces

### Suggesting Features

Feature requests are welcome! Please open an issue with:

- **Clear description** of the proposed feature
- **Use case** - why is this feature needed?
- **Examples** of how it would be used
- **Alternatives** you've considered

### Asking Questions

For questions about usage:
- Check the [documentation](HOW_TO_USE.md) first
- Search existing issues
- Open a new issue with the "question" label

## 🔧 Development Setup

### Prerequisites

- **C++ compiler** with C++17 support (GCC, Clang, or MSVC)
- **Python 3.8+** with pip
- **Rust 1.70+** (for CLI development)
- **Git**

### Setting Up Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/feather-db.git
cd feather-db

# 2. Build C++ core
g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o
ar rcs libfeather.a feather_core.o

# 3. Install Python dependencies
pip install pybind11 numpy

# 4. Build Python bindings
python setup.py build_ext --inplace

# 5. Install in development mode
pip install -e .

# 6. Build Rust CLI (optional)
cd feather-cli
cargo build --release
cd ..
```

### Running Tests

```bash
# Run Python examples
python3 examples/basic_python_example.py
python3 examples/semantic_search_example.py
python3 examples/batch_processing_example.py

# Run Rust CLI tests
./p-test/run_tests.sh

# Generate test data
python3 p-test/test_rust_cli.py
```

## 📝 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation if needed

### 3. Test Your Changes

- Ensure all existing tests pass
- Add new tests for new features
- Test on multiple platforms if possible
- Check for memory leaks (C++ changes)

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add feature: brief description"
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests when relevant

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- **Clear title** describing the change
- **Description** of what changed and why
- **Related issues** (if any)
- **Testing done** to verify the changes
- **Screenshots** (if applicable)

### 6. Code Review

- Respond to feedback promptly
- Make requested changes
- Keep the discussion focused and professional

## 💻 Code Style Guidelines

### Python

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable names
- Add docstrings to functions
- Type hints are encouraged

```python
def add_vector(db: feather_py.DB, id: int, vector: np.ndarray) -> None:
    """
    Add a vector to the database.
    
    Args:
        db: Feather database instance
        id: Unique identifier for the vector
        vector: NumPy array of floats
    """
    db.add(id=id, vec=vector)
```

### C++

- Use C++17 standard
- Follow existing naming conventions
- Use smart pointers for memory management
- Add comments for complex algorithms

```cpp
// Good: Clear naming and smart pointers
std::unique_ptr<DB> db = DB::open("db.feather", 768);

// Bad: Raw pointers and unclear names
DB* d = new DB();
```

### Rust

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- Follow Rust naming conventions
- Add documentation comments

```rust
/// Opens a database at the specified path
pub fn open(path: &Path, dim: usize) -> Option<Self> {
    // Implementation
}
```

## 🧪 Testing Guidelines

### Adding Tests

When adding new features:
1. Add unit tests for core functionality
2. Add integration tests for API changes
3. Update example code if relevant
4. Test edge cases and error conditions

### Test Coverage

- Aim for high test coverage
- Test both success and failure cases
- Test with different dimensions
- Test with large datasets

## 📚 Documentation Guidelines

### Updating Documentation

When making changes that affect users:
- Update relevant markdown files
- Add examples for new features
- Update API reference
- Keep documentation clear and concise

### Documentation Files

- `README.md` - Project overview and quick start
- `HOW_TO_USE.md` - Beginner-friendly guide
- `USAGE_GUIDE.md` - Complete API reference
- `examples/` - Working code examples
- `CHANGELOG.md` - Version history

## 🐛 Debugging Tips

### Python Issues

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check vector dimensions
print(f"Vector shape: {vector.shape}")
print(f"Database dimension: {db.dim()}")
```

### C++ Issues

```bash
# Compile with debug symbols
g++ -g -std=c++17 src/feather_core.cpp -o test

# Use valgrind for memory leaks
valgrind --leak-check=full ./test
```

### Rust Issues

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo run

# Check for common issues
cargo clippy
```

## 🌟 Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Add vector deletion functionality
- [ ] Support for cosine similarity
- [ ] Metadata storage with vectors
- [ ] Multi-threaded operations
- [ ] Python type stubs (.pyi files)

### Medium Priority
- [ ] Additional distance metrics (Manhattan, Hamming)
- [ ] Batch search operations
- [ ] Progress callbacks for long operations
- [ ] Compression for storage
- [ ] Python async API

### Low Priority
- [ ] Web API/REST interface
- [ ] Docker container
- [ ] Benchmarking suite
- [ ] Additional language bindings (Go, Java)
- [ ] GUI tool

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other unprofessional conduct

### Enforcement

Violations may result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: [your.email@example.com]

## 🎓 Learning Resources

### Vector Databases
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [Vector Database Basics](https://www.pinecone.io/learn/vector-database/)

### Development Tools
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [C++ Reference](https://en.cppreference.com/)

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions and ideas
- **Email**: [your.email@example.com]

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to Feather DB! 🚀
