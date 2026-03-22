#!/usr/bin/env bash
# Install JDK 17 (Adoptium Temurin) without sudo.
# Skips installation if java is already available.
set -euo pipefail

JAVA_VERSION="17"
INSTALL_DIR="${JAVA_HOME:-$HOME/.jdk/jdk-${JAVA_VERSION}}"

if command -v java &>/dev/null; then
    echo "[install_java] Java already installed: $(java -version 2>&1 | head -1)"
    exit 0
fi

echo "[install_java] Java not found. Installing JDK ${JAVA_VERSION} via install-jdk..."

# Ensure parent directory exists (install-jdk uses os.mkdir, not os.makedirs)
mkdir -p "$(dirname "${INSTALL_DIR}")"

# Use the install-jdk Python package (included in project dependencies)
JAVA_PATH=$(python -c "
import jdk
path = jdk.install('${JAVA_VERSION}', path='${INSTALL_DIR}')
print(path)
")

echo "[install_java] JDK installed at: ${JAVA_PATH}"
echo ""
echo "Add the following to your shell profile (.bashrc, .zshrc, etc.):"
echo ""
echo "  export JAVA_HOME=\"${JAVA_PATH}\""
echo "  export PATH=\"\${JAVA_HOME}/bin:\${PATH}\""
