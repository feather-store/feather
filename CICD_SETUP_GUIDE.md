# CI/CD Setup Guide for Feather DB

This guide explains how to set up automated testing and publishing for Feather DB using GitHub Actions.

## 🎯 What's Included

### 1. Automated Testing (`.github/workflows/test.yml`)
- Runs on every push and pull request
- Tests on Ubuntu and macOS
- Tests Python 3.8, 3.9, 3.10, 3.11, 3.12
- Builds C++ core and Python bindings
- Runs basic functionality tests

### 2. PyPI Publishing (`.github/workflows/publish-pypi.yml`)
- Automatically publishes to PyPI on release
- Builds wheels for multiple platforms
- Can be triggered manually
- Validates packages before publishing

### 3. GitHub Releases (`.github/workflows/release.yml`)
- Creates GitHub release on version tags
- Builds Rust CLI binaries for multiple platforms
- Attaches binaries to release
- Generates release notes

---

## 📋 Setup Steps

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `feather-db`
3. Description: "Fast, lightweight vector database for similarity search"
4. Choose **Public**
5. **Don't** initialize with README (you already have one)
6. Click "Create repository"

### Step 2: Push Code to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with CI/CD setup"

# Add remote (replace YOURUSERNAME)
git remote add origin https://github.com/YOURUSERNAME/feather-db.git

# Push
git branch -M main
git push -u origin main
```

### Step 3: Set Up PyPI API Token

#### 3.1 Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Create account and verify email

#### 3.2 Create API Token
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `feather-db-github-actions`
4. Scope: "Entire account" (or specific project after first upload)
5. Click "Add token"
6. **Copy the token** (starts with `pypi-`)

#### 3.3 Add Token to GitHub Secrets
1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI token
6. Click "Add secret"

### Step 4: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click "Actions" tab
3. If prompted, click "I understand my workflows, go ahead and enable them"

---

## 🚀 How to Use CI/CD

### Automated Testing

**Triggers automatically on:**
- Every push to `main` or `develop` branch
- Every pull request to `main`

**What it does:**
- Builds on Ubuntu and macOS
- Tests with Python 3.8-3.12
- Runs import and basic functionality tests
- Reports results in GitHub Actions tab

**View results:**
1. Go to repository → Actions tab
2. Click on the workflow run
3. See test results for each platform/Python version

### Publishing to PyPI

#### Option 1: Automatic (Recommended)

**Create a release:**
```bash
# Tag your version
git tag v0.1.0
git push origin v0.1.0
```

**Then on GitHub:**
1. Go to repository → Releases
2. Click "Draft a new release"
3. Choose tag: `v0.1.0`
4. Release title: `Feather DB v0.1.0`
5. Description: Copy from CHANGELOG.md
6. Click "Publish release"

**What happens:**
- GitHub Actions automatically builds wheels
- Uploads to PyPI
- Creates GitHub release with binaries

#### Option 2: Manual Trigger

1. Go to repository → Actions
2. Click "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select branch
5. Click "Run workflow"

### Creating Releases

**Automatic release on tag:**
```bash
# Create and push tag
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

**What happens:**
- Creates GitHub release
- Builds Rust CLI for Linux and macOS
- Attaches binaries to release
- Generates release notes

---

## 📊 Workflow Details

### Test Workflow

**File:** `.github/workflows/test.yml`

**Runs on:**
- Push to main/develop
- Pull requests to main

**Matrix:**
- OS: Ubuntu, macOS
- Python: 3.8, 3.9, 3.10, 3.11, 3.12

**Steps:**
1. Checkout code
2. Set up Python
3. Install dependencies
4. Build C++ core
5. Build Python bindings
6. Test import
7. Run basic tests

**Duration:** ~5-10 minutes per matrix job

### PyPI Publish Workflow

**File:** `.github/workflows/publish-pypi.yml`

**Runs on:**
- GitHub release published
- Manual trigger

**Steps:**
1. Build wheels for each platform/Python version
2. Upload artifacts
3. Publish to PyPI (on release only)

**Duration:** ~10-15 minutes

### Release Workflow

**File:** `.github/workflows/release.yml`

**Runs on:**
- Tags matching `v*` (e.g., v0.1.0)

**Steps:**
1. Create GitHub release
2. Build Rust CLI for multiple platforms
3. Attach binaries to release

**Platforms:**
- Linux x86_64
- macOS x86_64
- macOS ARM64 (M1/M2)

**Duration:** ~15-20 minutes

---

## 🔧 Customization

### Change Python Versions

Edit `.github/workflows/test.yml` and `.github/workflows/publish-pypi.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    # Add or remove versions as needed
```

### Add Windows Support

Add to matrix in workflows:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
```

**Note:** May need to adjust build commands for Windows.

### Change Release Trigger

Edit `.github/workflows/release.yml`:

```yaml
on:
  push:
    tags:
      - 'v*'  # Current: v0.1.0, v1.0.0, etc.
      # Or use: 'release-*' for release-0.1.0
```

### Add More Tests

Edit `.github/workflows/test.yml`:

```yaml
- name: Run comprehensive tests
  run: |
    python test_complete.py
```

---

## 🐛 Troubleshooting

### Tests Fail on GitHub but Pass Locally

**Common causes:**
1. **Different Python version** - Check matrix versions
2. **Missing dependencies** - Add to workflow
3. **Platform differences** - Test on Ubuntu/macOS locally

**Solution:**
```yaml
- name: Install system dependencies (Ubuntu)
  if: runner.os == 'Linux'
  run: |
    sudo apt-get update
    sudo apt-get install -y build-essential
```

### PyPI Upload Fails

**Error:** "Invalid or non-existent authentication"

**Solution:**
1. Check `PYPI_API_TOKEN` secret is set correctly
2. Verify token hasn't expired
3. Ensure token has correct scope

**Error:** "File already exists"

**Solution:**
- Can't re-upload same version
- Increment version number
- Or delete from PyPI (if within 24 hours)

### Rust CLI Build Fails

**Error:** "cargo: command not found"

**Solution:**
- Workflow already includes Rust installation
- Check `actions-rs/toolchain@v1` step

**Error:** "linking with `cc` failed"

**Solution:**
```yaml
- name: Install build tools
  run: |
    # Ubuntu
    sudo apt-get install -y build-essential
    # macOS (usually pre-installed)
```

### Workflow Doesn't Trigger

**Check:**
1. Actions are enabled (Settings → Actions)
2. Workflow file is in `.github/workflows/`
3. YAML syntax is correct
4. Branch/tag matches trigger conditions

---

## 📈 Monitoring

### View Workflow Status

**Badge in README:**
```markdown
![Tests](https://github.com/YOURUSERNAME/feather-db/workflows/Tests/badge.svg)
![PyPI](https://img.shields.io/pypi/v/feather-db)
```

**GitHub Actions Tab:**
- See all workflow runs
- View logs for each step
- Download artifacts
- Re-run failed jobs

### Email Notifications

GitHub sends emails for:
- Failed workflows
- Successful releases
- Can configure in Settings → Notifications

---

## 🎯 Best Practices

### 1. Test Before Release

```bash
# Run tests locally
python test_complete.py

# Create release candidate tag
git tag v0.1.0-rc1
git push origin v0.1.0-rc1

# Check CI passes
# Then create final release
git tag v0.1.0
git push origin v0.1.0
```

### 2. Semantic Versioning

- **v0.1.0** - Initial release
- **v0.1.1** - Bug fixes
- **v0.2.0** - New features
- **v1.0.0** - Stable release

### 3. Changelog

Update `CHANGELOG.md` before each release:

```markdown
## [0.1.1] - 2025-11-20

### Fixed
- Bug in search function
- Memory leak in C++ core

### Added
- Support for Python 3.13
```

### 4. Pre-release Testing

Use TestPyPI for testing:

```yaml
- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}
    repository-url: https://test.pypi.org/legacy/
```

---

## 📋 Release Checklist

Before creating a release:

- [ ] All tests pass locally
- [ ] Version number updated in:
  - [ ] `setup.py`
  - [ ] `pyproject.toml`
  - [ ] `feather-cli/Cargo.toml`
  - [ ] `CHANGELOG.md`
- [ ] CHANGELOG.md updated
- [ ] Documentation updated
- [ ] Examples tested
- [ ] Committed and pushed to main
- [ ] CI tests pass on GitHub

Create release:

- [ ] Create and push tag: `git tag v0.1.0 && git push origin v0.1.0`
- [ ] Create GitHub release
- [ ] Wait for CI to complete
- [ ] Verify package on PyPI
- [ ] Test installation: `pip install feather-db`
- [ ] Announce release

---

## 🚀 Quick Start Commands

```bash
# 1. Push code to GitHub
git init
git add .
git commit -m "Initial commit with CI/CD"
git remote add origin https://github.com/YOURUSERNAME/feather-db.git
git push -u origin main

# 2. Create release
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# 3. Create GitHub release (via web interface)
# Go to: https://github.com/YOURUSERNAME/feather-db/releases/new

# 4. Wait for CI/CD to complete

# 5. Verify
pip install feather-db
python -c "import feather_py; print('Success!')"
```

---

## 📞 Support

### GitHub Actions Documentation
- https://docs.github.com/en/actions

### PyPI Publishing
- https://packaging.python.org/
- https://pypi.org/help/

### Troubleshooting
- Check workflow logs in Actions tab
- Review error messages
- Test locally first
- Ask in GitHub Discussions

---

## ✨ Summary

With this CI/CD setup:

✅ **Automated testing** on every push  
✅ **Automatic PyPI publishing** on release  
✅ **Multi-platform builds** for Rust CLI  
✅ **GitHub releases** with binaries  
✅ **Professional workflow** like top libraries  

Your library will be published automatically whenever you create a new release tag!

**Next step:** Push to GitHub and create your first release! 🚀
