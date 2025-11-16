# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-16

### Added
- Initial release of Feather DB
- Python API with NumPy integration and pybind11 bindings
- C++ core implementation with HNSW algorithm
- Rust CLI for command-line operations
- Binary file format with magic number validation for persistence
- Support for L2 (Euclidean) distance metric
- SIMD optimizations (AVX512/AVX/SSE) for distance calculations
- Comprehensive documentation:
  - HOW_TO_USE.md - Beginner-friendly guide
  - USAGE_GUIDE.md - Complete API reference
  - Architecture diagrams and internals documentation
- Working examples for common use cases:
  - Basic Python usage
  - Semantic search implementation
  - Batch processing for large datasets
- Batch processing capabilities with periodic saves
- Configurable k parameter for search results
- Automatic database save on destruction
- Memory-efficient vector storage

### Features
- **Fast Search**: Approximate nearest neighbor search using HNSW algorithm
- **Multi-Language**: Python, C++, and Rust APIs
- **Persistent Storage**: Custom binary format with header validation
- **Scalable**: Supports up to 1 million vectors (configurable)
- **Easy to Use**: Simple, intuitive APIs across all languages
- **Production Ready**: Tested with comprehensive test suite

### Performance
- Add rate: 2,000-5,000 vectors/second (depending on dimension)
- Search time: 0.5-1.5ms per query (k=10)
- Memory usage: ~4 bytes per dimension per vector + index overhead
- Tested with up to 10,000 vectors

### Documentation
- Complete usage guide with real-world examples
- Beginner-friendly how-to guide
- Architecture documentation with visual diagrams
- Performance benchmarks and optimization tips
- Troubleshooting guide
- API reference for all three languages

### Testing
- Automated test suite for Rust CLI
- Test data generation scripts
- Validation of search accuracy
- Binary format verification
- Memory leak testing

### Known Limitations
- Maximum 1 million vectors (configurable in C++ code)
- Only L2 distance metric supported
- No vector deletion functionality
- No metadata storage with vectors
- Single-threaded operations

[Unreleased]: https://github.com/yourusername/feather-db/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/feather-db/releases/tag/v0.1.0
