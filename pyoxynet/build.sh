#!/bin/bash
set -e  # Exit on any error

echo "🚀 Pyoxynet Build & Publishing Script"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "setup.py" ]]; then
    print_error "setup.py not found! Make sure you're in the pyoxynet package directory."
    exit 1
fi

print_status "Starting build process for pyoxynet..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d" " -f2 | cut -d"." -f1-2)
print_status "Using Python $PYTHON_VERSION"

# Step 1: Clean previous builds
print_status "Cleaning previous build artifacts..."
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/
print_success "Cleaned build artifacts"

# Step 2: Install/upgrade build tools
print_status "Ensuring build tools are up to date..."
python3 -m pip install --upgrade build twine
print_success "Build tools updated"

# Step 3: Run tests (if available)
print_status "Running tests..."
if [[ -f "../test_lightweight_final.py" ]]; then
    cd ..
    python3 test_lightweight_final.py
    cd pyoxynet
    print_success "Tests passed"
else
    print_warning "No tests found, skipping test phase"
fi

# Step 4: Build the package
print_status "Building wheel and source distribution..."
python3 -m build
print_success "Package built successfully"

# Step 5: Check the distribution
print_status "Checking package integrity..."
python3 -m twine check dist/*
print_success "Package integrity verified"

# Step 6: Show package info
print_status "Package contents:"
ls -la dist/

# Step 7: Ask for confirmation before upload
echo ""
print_warning "Ready to upload to PyPI!"
print_status "Package versions in dist/:"
ls dist/

read -p "Do you want to upload to PyPI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Uploading to PyPI..."
    python3 -m twine upload dist/*
    print_success "Package uploaded to PyPI!"
else
    print_status "Upload cancelled. You can upload manually later with:"
    print_status "python3 -m twine upload dist/*"
fi

# Step 8: Update documentation (if docs exist)
if [[ -d "../docs" ]]; then
    print_status "Updating documentation..."
    cd ../docs
    if command -v make &> /dev/null; then
        make html
        print_success "Documentation updated"
    else
        print_warning "Make command not found, skipping docs"
    fi
    cd ../pyoxynet
fi

print_success "Build process completed!"
echo ""
print_status "Next steps:"
echo "  1. Check your package on PyPI: https://pypi.org/project/pyoxynet/"
echo "  2. Test installation: pip install --upgrade pyoxynet"
echo "  3. Test full installation: pip install --upgrade pyoxynet[full]"