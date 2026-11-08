"""Native-feeling app window and desktop launchers, with no dependencies.

open_window(): opens Lens in its own Chromium-family app window (Chrome, Edge, Chromium, Brave): no tabs or
address bar, its own taskbar entry. Falls back to the default browser.
install_launcher(): adds "Proprioceptive Lens" to the desktop menu (Linux .desktop, macOS ~/Applications app,
Windows Start-menu shortcut via a .bat). Returns the path written.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ICON = Path(__file__).resolve().parent / "static" / "icon.svg"
CANDIDATES = {
    "linux": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"],
    "darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
               "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
               "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
               "/Applications/Chromium.app/Contents/MacOS/Chromium"],
    "win32": [r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
              r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
              r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
              r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
              r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"],
}


def find_browser() -> str | None:
    platform = "win32" if sys.platform.startswith("win") else ("darwin" if sys.platform == "darwin" else "linux")
    for name in CANDIDATES[platform]:
        path = os.path.expandvars(name)
        found = shutil.which(path) if os.sep not in path else (path if os.path.exists(path) else None)
        if found:
            return found
    return None


def open_window(url: str, data_dir: Path) -> str:
    """Open Lens as an app window; returns 'window' or 'browser'."""
    browser = find_browser()
    if browser:
        profile = Path(data_dir) / "window-profile"          # separate profile = its own window and taskbar entry
        try:
            subprocess.Popen([browser, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1360,860",
                              "--no-first-run", "--no-default-browser-check"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return "window"
        except OSError:
            pass
    webbrowser.open(url)
    return "browser"


def install_launcher() -> Path:
    lens = shutil.which("lens") or f"{sys.executable} -m lens"
    if sys.platform.startswith("win"):
        folder = Path(os.environ.get("APPDATA", "~")).expanduser() / "Microsoft/Windows/Start Menu/Programs"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "Proprioceptive Lens.bat"
        target.write_text(f'@echo off\r\nstart "" /min {lens}\r\n')
        return target
    if sys.platform == "darwin":
        app = Path("~/Applications/Proprioceptive Lens.app/Contents/MacOS").expanduser()
        app.mkdir(parents=True, exist_ok=True)
        target = app / "Proprioceptive Lens"
        target.write_text(f"#!/bin/sh\nexec {lens}\n")
        target.chmod(0o755)
        return target.parents[2]
    folder = Path("~/.local/share/applications").expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "proprioceptive-lens.desktop"
    target.write_text("[Desktop Entry]\nType=Application\nName=Proprioceptive Lens\n"
                      "Comment=See inside your local AI\n"
                      f"Exec={lens}\nIcon={ICON}\nTerminal=false\nCategories=Development;Utility;\n")
    target.chmod(0o755)
    return target
