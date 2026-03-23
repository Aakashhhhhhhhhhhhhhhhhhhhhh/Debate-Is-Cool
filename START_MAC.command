#!/bin/bash
clear
echo ""
echo "  ===================================================="
echo "    DEBATE IS COOL — Monitoring System"
echo "    Civic Space Nepal Pvt. Ltd."
echo "  ===================================================="
echo ""

# ── Find Python ───────────────────────────────────────────
PY=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1)
        # Must be Python 3.8+
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)")
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
            PY="$cmd"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "  [!] Python 3.8 or higher is NOT installed."
    echo ""
    echo "  Please follow these steps:"
    echo ""
    echo "  1. Open your browser and go to:"
    echo "     https://www.python.org/downloads/"
    echo ""
    echo "  2. Download and install the latest Python for Mac"
    echo ""
    echo "  3. After installation, double-click this file again."
    echo ""
    # Try to open the download page on Mac
    if command -v open &>/dev/null; then
        open "https://www.python.org/downloads/"
    fi
    read -p "  Press Enter to close..."
    exit 1
fi

echo "  [OK] Found: $($PY --version)"
echo ""
echo "  Starting Debate is Cool..."
echo "  (First launch may take 1-2 minutes to install packages)"
echo ""

cd "$(dirname "$0")"
$PY launcher.py

if [ $? -ne 0 ]; then
    echo ""
    echo "  [!] Something went wrong."
    echo "  Please send a screenshot of this window to your administrator."
    read -p "  Press Enter to close..."
fi
