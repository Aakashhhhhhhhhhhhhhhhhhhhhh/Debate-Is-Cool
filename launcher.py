"""
DEBATE IS COOL — Launcher | Civic Space Nepal Pvt. Ltd.
Works on any device with any Python 3.8+ version.
Automatically installs compatible package versions.
"""
import sys, os, subprocess, time, threading, socket, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
APP  = os.path.join(HERE, "app.py")

def install_packages():
    pip = [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"]

    # Step 1: Upgrade pip itself first
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 2: Install numpy and pandas together so they are always compatible
    print("  Installing core packages (numpy, pandas) ...")
    subprocess.call(pip + ["numpy", "pandas"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 3: Install everything else at latest versions
    print("  Installing remaining packages ...")
    others = [
        "streamlit", "plotly", "scikit-learn", "joblib",
        "pymongo", "fpdf2", "gspread",
        "google-auth", "google-auth-oauthlib",
    ]
    subprocess.call(pip + others,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("  All packages ready.")

def _free_port(preferred=8501):
    for p in [preferred] + list(range(8502, 8600)):
        try:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("localhost", p))
            s.close()
            return p
        except OSError:
            continue
    return preferred

def _open_browser(port):
    url = f"http://localhost:{port}"
    for _ in range(120):
        time.sleep(0.5)
        try:
            conn = socket.create_connection(("localhost", port), timeout=1)
            conn.close()
            time.sleep(1.0)
            webbrowser.open(url)
            print(f"\n  Browser opened: {url}")
            print(f"  If it did not open, paste this in Chrome/Edge:\n  {url}\n")
            return
        except (OSError, ConnectionRefusedError):
            continue
    print(f"\n  Please open manually in browser: http://localhost:{port}\n")

def main():
    print()
    print("  ==================================================")
    print("  DEBATE IS COOL - Monitoring System")
    print("  Civic Space Nepal Pvt. Ltd.")
    print("  ==================================================")
    print()

    if not os.path.isfile(APP):
        print(f"  ERROR: app.py not found at:\n  {APP}")
        print("\n  Make sure launcher.py and app.py are in the SAME folder.")
        input("\n  Press Enter to close...")
        return

    print("  [1/3] Setting up packages ...")
    print("  (First run takes 2-3 min — please wait)")
    install_packages()

    port = _free_port(8501)
    url  = f"http://localhost:{port}"

    print(f"\n  [2/3] Starting server ...")
    print()
    print("  --------------------------------------------------")
    print(f"  App URL :  {url}")
    print("  --------------------------------------------------")
    print("  Browser will open automatically.")
    print("  If not, copy the URL above into Chrome or Edge.")
    print()
    print("  Keep this window open while using the app.")
    print("  Close this window to stop the app.")
    print()
    print("  [3/3] Running ...")
    print()

    t = threading.Thread(target=_open_browser, args=(port,), daemon=True)
    t.start()

    result = subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        APP,
        "--server.port",              str(port),
        "--server.headless",          "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType",   "none",
        "--server.address",           "localhost",
    ])

    if result.returncode != 0:
        print("\n  Something went wrong.")
        print(f"  app.py path: {APP}")
        input("  Press Enter to close...")

if __name__ == "__main__":
    main()
