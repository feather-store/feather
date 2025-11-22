#!/bin/bash

# Feather DB - CI/CD Setup Script
# This script helps you set up GitHub Actions for automated publishing

set -e

echo "🚀 Feather DB - CI/CD Setup"
echo "============================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if .github/workflows exists
if [ ! -d ".github/workflows" ]; then
    echo -e "${YELLOW}Creating .github/workflows directory...${NC}"
    mkdir -p .github/workflows
fi

echo -e "${GREEN}✓ GitHub Actions workflows are ready${NC}"
echo ""

# Check git
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}✓ Git initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository exists${NC}"
fi
echo ""

# Check for GitHub remote
if ! git remote get-url origin &> /dev/null; then
    echo -e "${YELLOW}⚠️  No GitHub remote configured${NC}"
    echo ""
    echo "To add GitHub remote:"
    echo -e "${BLUE}  git remote add origin https://github.com/YOURUSERNAME/feather-db.git${NC}"
    echo ""
else
    REMOTE=$(git remote get-url origin)
    echo -e "${GREEN}✓ GitHub remote: ${REMOTE}${NC}"
    echo ""
fi

# Instructions
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Create GitHub repository:"
echo -e "   ${BLUE}https://github.com/new${NC}"
echo ""
echo "2. Push code to GitHub:"
echo -e "   ${BLUE}git add .${NC}"
echo -e "   ${BLUE}git commit -m \"Add CI/CD workflows\"${NC}"
echo -e "   ${BLUE}git push -u origin main${NC}"
echo ""
echo "3. Set up PyPI token:"
echo -e "   a. Get token: ${BLUE}https://pypi.org/manage/account/token/${NC}"
echo -e "   b. Add to GitHub: ${BLUE}Settings → Secrets → Actions${NC}"
echo -e "   c. Name: ${BLUE}PYPI_API_TOKEN${NC}"
echo ""
echo "4. (Optional) Set up Cargo token:"
echo -e "   a. Get token: ${BLUE}https://crates.io/settings/tokens${NC}"
echo -e "   b. Add to GitHub: ${BLUE}Settings → Secrets → Actions${NC}"
echo -e "   c. Name: ${BLUE}CARGO_TOKEN${NC}"
echo ""
echo "5. Create a release:"
echo -e "   ${BLUE}git tag v0.1.0${NC}"
echo -e "   ${BLUE}git push origin v0.1.0${NC}"
echo ""
echo "6. Watch the magic happen! ✨"
echo -e "   ${BLUE}https://github.com/YOURUSERNAME/feather-db/actions${NC}"
echo ""
echo "📖 For detailed instructions, see: ${BLUE}CICD_SETUP_GUIDE.md${NC}"
echo ""
echo -e "${GREEN}🎉 CI/CD setup complete!${NC}"
