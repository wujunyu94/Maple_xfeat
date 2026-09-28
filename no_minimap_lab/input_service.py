"""Task-scoped elevated keyboard bridge; no shell or arbitrary command support."""
import ctypes
import json
import os
from pathlib import Path
import time

from src.core.input_driver import InputDriver
from src.core.window import WindowManager

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "output/navigation_control"
ALLOWED = {"left", "right", "up", "down", "alt", "tab"}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    wm = WindowManager()
    if not wm.find_game_window():
        raise RuntimeError("Game window missing")
    driver = InputDriver(wm.hwnd, input_mode="foreground")
    ready, reason = driver.check_input_readiness()
    write_json(STATE / "service.json", dict(pid=os.getpid(), hwnd=wm.hwnd, game_pid=wm.pid,
        admin=bool(ctypes.windll.shell32.IsUserAnAdmin()), ready=ready, reason=reason))
    if not ready:
        return
    started = time.time()
    sequence = None
    expires = 0
    held = set()
    audit = (STATE / "keys.jsonl").open("a", encoding="utf-8", buffering=1)
    try:
        while time.time()-started < 12*3600:
            now = time.time()
            if ctypes.windll.user32.GetAsyncKeyState(0x7B) & 0x8000:  # F12
                break
            try:
                command = json.loads((STATE / "command.json").read_text(encoding="utf-8"))
                if command.get("sequence") != sequence and abs(now-float(command["time"])) < 2:
                    sequence = command["sequence"]
                    if command.get("stop"):
                        break
                    desired = set(command.get("keys", []))
                    if not desired <= ALLOWED or {"left", "right"} <= desired:
                        raise ValueError("Disallowed keys")
                    expires = min(now+0.5, float(command["time"])+float(command.get("ttl", .3)))
                    if desired and not driver.ensure_focus():
                        desired = set()
                    for key in held-desired:
                        driver.key_up(key)
                    for key in desired-held:
                        driver.key_down(key)
                    if held != desired:
                        audit.write(json.dumps(dict(time=now, monotonic=time.perf_counter(), sequence=sequence,
                                                    reason='command', keys=sorted(desired)))+"\n")
                    held = desired
                    write_json(STATE / "ack.json", dict(sequence=sequence, time=now, keys=sorted(held)))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if now > expires or not wm.is_valid() or ctypes.windll.user32.GetForegroundWindow() != wm.hwnd:
                if held:
                    audit.write(json.dumps(dict(time=now, monotonic=time.perf_counter(), sequence=sequence,
                        reason='expired' if now > expires else 'window_or_focus_lost', keys=[]))+"\n")
                for key in held:
                    driver.key_up(key)
                held = set()
            time.sleep(.01)
    finally:
        driver.release_all_keys()
        audit.close()
        write_json(STATE / "service.json", dict(pid=os.getpid(), stopped=True, time=time.time()))


if __name__ == "__main__":
    main()
