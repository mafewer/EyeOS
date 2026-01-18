import os
import cv2
import mediapipe as mp
import pyautogui
import time
import math
import threading
from collections import deque
import customtkinter as ctk
from PIL import Image
import tkinter as tk
import tkinter.filedialog as fd

# Optional macOS overlay helpers (only used if PyObjC is available)
try:
    from AppKit import NSApplication  # type: ignore
    from AppKit import (
        NSWindowCollectionBehaviorCanJoinAllSpaces,  # type: ignore
        NSWindowCollectionBehaviorTransient,  # type: ignore
        NSWindowCollectionBehaviorFullScreenAuxiliary,  # type: ignore
        NSStatusWindowLevel,  # type: ignore
    )
    _HAS_PYOBJC = True
except Exception:
    _HAS_PYOBJC = False
import global_var
import utilities

# ------------------- GLOBALS -------------------
pyautogui.FAILSAFE = False
screen_width, screen_height = pyautogui.size()
isSettingsOpen = False

cap = None
tracking_active = threading.Event()
stop_event = threading.Event()

mp_face_mesh = mp.solutions.face_mesh # pyright: ignore[reportAttributeAccessIssue]
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Eye indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Settings (adjustable by UI)
EAR_THRESHOLD_LEFT = 0.22
EAR_THRESHOLD_RIGHT = 0.22
MOVEMENT_GAIN = 1.0

MIN_CONSEC_FRAMES = 2
CLICK_COOLDOWN = 0.5


# ------------------- DWELL (GAZE HOLD) CLICK SETTINGS -------------------
USE_DWELL_CLICK = True          # Proof-of-concept: dwell-to-left-click
USE_BLINK_CLICK = False         # Set True if you want to keep blink clicks enabled

DWELL_TIME_SEC = 1.2            # How long you must hold gaze to click
DWELL_RADIUS_PX = 45            # How far the cursor can drift while "holding"
DWELL_ARM_DELAY_SEC = 0.15      # Small delay before dwell starts counting (reduces accidental starts)
DWELL_COOLDOWN_SEC = 0.6        # Prevent immediate double-clicks

# Cursor smoothing (EMA)
CURSOR_SMOOTH_ALPHA = 0.25      # 0..1 (higher = snappier, lower = smoother)

# Test mode: if camera/face isn't available, use the real mouse position as the "gaze cursor"
TEST_MODE_MOUSE_FALLBACK = True
TEST_MODE_TICK_SEC = 0.01

# ------------------- DWELL INDICATOR (PROGRESS BAR OVERLAY) -------------------
# A small bar that follows the cursor and fills based on _dwell_progress.
SHOW_DWELL_BAR = True
DWELL_BAR_UPDATE_MS = 33        # ~30 FPS, low overhead
DWELL_BAR_W = 64                # px
DWELL_BAR_H = 8                 # px
DWELL_BAR_BORDER = 1            # px
DWELL_BAR_OFFSET_X = 0          # px (centered under cursor)
DWELL_BAR_OFFSET_Y = 22         # px (below cursor)
DWELL_BAR_HIDE_WHEN_IDLE = True
DWELL_BAR_ALPHA_FALLBACK = 0.55 # used if transparent color isn't supported

DWELL_BAR_BG = "#ff00ff"         # transparent key color (magenta)
DWELL_BAR_OUTLINE = "#3a3a3a"    # subtle outline
DWELL_BAR_FILL = "#00E5FF"       # bright cyan fill
DWELL_BAR_EMPTY = "#111111"      # background in fallback mode

_dwellbar_win = None
_dwellbar_canvas = None
_dwellbar_fill_rect = None
_dwellbar_outline_rect = None
_dwellbar_using_transparent = False

# Dwell state
_dwell_candidate = None         # (x, y) anchor for dwell
_dwell_arm_start = 0.0
_dwell_start = 0.0
_dwell_cooldown_until = 0.0
_dwell_progress = 0.0
_last_progress_print = 0.0

# Smoothed cursor position
_sx = None
_sy = None

# Blink counters
left_counter = 0
right_counter = 0
last_left_click = 0
last_right_click = 0

# EAR smoothing
ear_queue_left = deque(maxlen=5)
ear_queue_right = deque(maxlen=5)


# ------------------- HELPERS -------------------
def euclidean(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def get_ear(landmarks, indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def ema(prev, current, alpha):
    if prev is None:
        return current
    return prev + alpha * (current - prev)


def _configure_macos_overlay(tk_toplevel: tk.Toplevel):
    """macOS: show overlay on all Spaces + over full-screen, and make it click-through."""
    if not _HAS_PYOBJC:
        return

    try:
        app = NSApplication.sharedApplication()
        try:
            tk_toplevel.update_idletasks()
        except Exception:
            pass

        wins = list(app.windows())
        if not wins:
            return

        # Try to match the correct native window via title
        desired_title = ""
        try:
            desired_title = tk_toplevel.title()
        except Exception:
            desired_title = ""

        target = None
        if desired_title:
            for w in reversed(wins):
                try:
                    if str(w.title()) == str(desired_title):
                        target = w
                        break
                except Exception:
                    continue

        if target is None:
            # Fallback: last window
            target = wins[-1]

        # Always-on-top
        try:
            target.setLevel_(NSStatusWindowLevel)
        except Exception:
            pass

        # Click-through
        try:
            target.setIgnoresMouseEvents_(True)
        except Exception:
            pass

        # Key: show on all Spaces + over full-screen
        try:
            behavior = (
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
                | NSWindowCollectionBehaviorTransient
            )
            target.setCollectionBehavior_(behavior)
        except Exception:
            pass

        # Don’t hide when app deactivates
        try:
            target.setHidesOnDeactivate_(False)
        except Exception:
            pass

    except Exception:
        return


def dwell_update_and_maybe_click(cur_x, cur_y, now):
    """Update dwell state from a cursor position (gaze OR mouse fallback). Returns dwell progress (0..1)."""
    global _dwell_candidate, _dwell_arm_start, _dwell_start, _dwell_cooldown_until, _dwell_progress

    if not USE_DWELL_CLICK:
        _dwell_progress = 0.0
        return _dwell_progress

    # Respect cooldown
    if now < _dwell_cooldown_until:
        _dwell_progress = 0.0
        return _dwell_progress

    cur = (int(cur_x), int(cur_y))

    # If no candidate yet, (re)arm on current position
    if _dwell_candidate is None:
        _dwell_candidate = cur
        _dwell_arm_start = now
        _dwell_start = 0.0
        _dwell_progress = 0.0
        return _dwell_progress

    # If cursor wandered too far, reset candidate
    if dist2(cur, _dwell_candidate) > (DWELL_RADIUS_PX * DWELL_RADIUS_PX):
        _dwell_candidate = cur
        _dwell_arm_start = now
        _dwell_start = 0.0
        _dwell_progress = 0.0
        return _dwell_progress

    # Within radius: after a short arm delay, start counting dwell time
    if _dwell_start == 0.0:
        if (now - _dwell_arm_start) >= DWELL_ARM_DELAY_SEC:
            _dwell_start = now
            _dwell_progress = 0.0
        return _dwell_progress

    elapsed = now - _dwell_start
    _dwell_progress = max(0.0, min(1.0, elapsed / DWELL_TIME_SEC))

    if elapsed >= DWELL_TIME_SEC:
        pyautogui.click(button="left")
        print("Dwell → LEFT CLICK")
        _dwell_cooldown_until = now + DWELL_COOLDOWN_SEC
        _dwell_candidate = None
        _dwell_arm_start = 0.0
        _dwell_start = 0.0
        _dwell_progress = 0.0

    return _dwell_progress


# ------------------- DWELL PROGRESS BAR (UI OVERLAY) -------------------
def init_dwell_bar_overlay(parent):
    """Create a tiny always-on-top overlay window that shows dwell progress as a loading bar."""
    global _dwellbar_win, _dwellbar_canvas, _dwellbar_fill_rect, _dwellbar_outline_rect, _dwellbar_using_transparent

    if not SHOW_DWELL_BAR:
        return

    if _dwellbar_win is not None:
        return

    win = tk.Toplevel(parent)
    win.overrideredirect(True)
    win.attributes("-topmost", True)

    # Unique title so we can locate the native NSWindow reliably
    try:
        win.title("EyeOS_DwellBarOverlay")
    except Exception:
        pass

    # macOS: keep overlay on top of full-screen apps and ignore mouse events
    _configure_macos_overlay(win)

    # Retry once shortly after creation (native NSWindow can appear a tick later)
    try:
        parent.after(200, lambda: _configure_macos_overlay(win))
    except Exception:
        pass
    # Never steal focus (critical on macOS)
    try:
        win.attributes("-takefocus", 0)
    except Exception:
        pass

    # Prefer true transparency via transparent color; fall back to alpha with a dark background.
    _dwellbar_using_transparent = False
    try:
        win.configure(bg=DWELL_BAR_BG)
        win.wm_attributes("-transparentcolor", DWELL_BAR_BG)
        _dwellbar_using_transparent = True
    except Exception:
        win.configure(bg=DWELL_BAR_EMPTY)
        try:
            win.attributes("-alpha", DWELL_BAR_ALPHA_FALLBACK)
        except Exception:
            pass

    win.geometry(f"{DWELL_BAR_W}x{DWELL_BAR_H}+0+0")

    canvas = tk.Canvas(
        win,
        width=DWELL_BAR_W,
        height=DWELL_BAR_H,
        highlightthickness=0,
        bd=0,
        bg=DWELL_BAR_BG if _dwellbar_using_transparent else win["bg"],
    )
    canvas.pack(fill="both", expand=True)

    x0 = DWELL_BAR_BORDER
    y0 = DWELL_BAR_BORDER
    x1 = DWELL_BAR_W - DWELL_BAR_BORDER
    y1 = DWELL_BAR_H - DWELL_BAR_BORDER

    # Outline
    outline = canvas.create_rectangle(
        x0, y0, x1, y1,
        outline=DWELL_BAR_OUTLINE,
        width=1,
    )

    # Fill (starts empty)
    fill = canvas.create_rectangle(
        x0, y0, x0, y1,
        outline="",
        fill=DWELL_BAR_FILL,
        width=0,
    )

    _dwellbar_win = win
    _dwellbar_canvas = canvas
    _dwellbar_outline_rect = outline
    _dwellbar_fill_rect = fill

    # Start fully transparent (avoid withdraw/deiconify which can trigger Space switches on macOS)
    try:
        win.attributes("-alpha", 0.0)
    except Exception:
        pass


def update_dwell_bar_overlay():
    """Update bar position/fill based on _dwell_progress. Scheduled via root.after()."""
    global _dwellbar_win, _dwellbar_canvas, _dwellbar_fill_rect, _dwellbar_outline_rect

    if not SHOW_DWELL_BAR:
        return

    if _dwellbar_win is None or _dwellbar_canvas is None or _dwellbar_fill_rect is None:
        return

    try:
        active = (_dwell_candidate is not None) or (_dwell_progress > 0.0)

        # Follow cursor (uses OS cursor position; works in both gaze mode and test mode)
        cx, cy = pyautogui.position()
        x = int(cx - (DWELL_BAR_W // 2) + DWELL_BAR_OFFSET_X)
        y = int(cy + DWELL_BAR_OFFSET_Y)
        _dwellbar_win.geometry(f"{DWELL_BAR_W}x{DWELL_BAR_H}+{x}+{y}")

        # Fade instead of withdraw/deiconify to avoid macOS Space switches
        if DWELL_BAR_HIDE_WHEN_IDLE and not active:
            try:
                _dwellbar_win.attributes("-alpha", 0.0)
            except Exception:
                pass
        else:
            try:
                # If transparentcolor worked, use fully opaque; otherwise use the fallback alpha
                alpha = 1.0 if _dwellbar_using_transparent else DWELL_BAR_ALPHA_FALLBACK
                _dwellbar_win.attributes("-alpha", alpha)
            except Exception:
                pass

        # Compute fill width
        p = max(0.0, min(1.0, float(_dwell_progress)))
        x0 = DWELL_BAR_BORDER
        y0 = DWELL_BAR_BORDER
        x1 = DWELL_BAR_W - DWELL_BAR_BORDER
        y1 = DWELL_BAR_H - DWELL_BAR_BORDER
        fill_x1 = int(x0 + p * (x1 - x0))

        _dwellbar_canvas.coords(_dwellbar_fill_rect, x0, y0, fill_x1, y1)

        # Subtle outline brightening when progress is active
        if _dwell_progress > 0.0:
            _dwellbar_canvas.itemconfigure(_dwellbar_outline_rect, outline="#5a5a5a")
        else:
            _dwellbar_canvas.itemconfigure(_dwellbar_outline_rect, outline=DWELL_BAR_OUTLINE)

        # Reschedule (non-blocking)
        root.after(DWELL_BAR_UPDATE_MS, update_dwell_bar_overlay)
    except Exception:
        return


# ------------------- TRACKING LOOP -------------------
def tracking_loop():
    global left_counter, right_counter, last_left_click, last_right_click
    global EAR_THRESHOLD_LEFT, EAR_THRESHOLD_RIGHT, MOVEMENT_GAIN
    global _dwell_candidate, _dwell_arm_start, _dwell_start, _dwell_cooldown_until, _dwell_progress, _last_progress_print
    global _sx, _sy

    global cap
    cap = cv2.VideoCapture(utilities.get_camera_input())
    while not stop_event.is_set():

        if global_var.camera_input_changed:
            global_var.camera_input_changed = False
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(utilities.get_camera_input())

        if cap is None or not cap.isOpened():
            # Test mode fallback: allow dwell-click testing using the physical mouse
            if TEST_MODE_MOUSE_FALLBACK and tracking_active.is_set():
                now = time.time()
                mx, my = pyautogui.position()
                dwell_update_and_maybe_click(mx, my, now)

                if now - _last_progress_print > 0.5:
                    _last_progress_print = now
                    print(f"[TEST MODE] Dwell progress: {_dwell_progress:.2f}")

                time.sleep(TEST_MODE_TICK_SEC)
            else:
                time.sleep(0.05)
            continue

        if not tracking_active.is_set():
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
    
            # Cursor movement
            left_center = landmarks[473]
            right_center = landmarks[468]
            eye_x = (left_center.x + right_center.x) / 2
            eye_y = (left_center.y + right_center.y) / 2

            x_range = (0.375, 0.625)
            y_range = (0.375, 0.625)
            eye_x = max(min(eye_x, x_range[1]), x_range[0])
            eye_y = max(min(eye_y, y_range[1]), y_range[0])
            norm_x = (eye_x - x_range[0]) / (x_range[1] - x_range[0])
            norm_y = (eye_y - y_range[0]) / (y_range[1] - y_range[0])

            gain = max(0.1, min(2.0, MOVEMENT_GAIN))
            target_x = int(norm_x * screen_width * gain)
            target_y = int(norm_y * screen_height * gain)

            # Smooth the cursor to reduce jitter before dwell logic
            _sx = ema(_sx, float(target_x), CURSOR_SMOOTH_ALPHA)
            _sy = ema(_sy, float(target_y), CURSOR_SMOOTH_ALPHA)
            sx_i = int(_sx)
            sy_i = int(_sy)

            pyautogui.moveTo(sx_i, sy_i)

            # ------------------- DWELL (GAZE HOLD) CLICK -------------------
            now = time.time()
            dwell_update_and_maybe_click(sx_i, sy_i, now)

            # Light debug print (twice per second max)
            if now - _last_progress_print > 0.5:
                _last_progress_print = now
                if _dwell_candidate is not None:
                    print(f"Dwell progress: {_dwell_progress:.2f}")

            if USE_BLINK_CLICK:
                # EAR blink detection
                ear_left = get_ear(landmarks, LEFT_EYE)
                ear_right = get_ear(landmarks, RIGHT_EYE)
                ear_queue_left.append(ear_left)
                ear_queue_right.append(ear_right)
                avg_ear_left = sum(ear_queue_left) / len(ear_queue_left)
                avg_ear_right = sum(ear_queue_right) / len(ear_queue_right)

                # Left blink
                if avg_ear_left < EAR_THRESHOLD_LEFT:
                    left_counter += 1
                else:
                    if left_counter >= MIN_CONSEC_FRAMES and now - last_left_click > CLICK_COOLDOWN:
                        pyautogui.click(button="left")
                        print("Left blink → LEFT CLICK")
                        last_left_click = now
                    left_counter = 0

                # Right blink
                if avg_ear_right < EAR_THRESHOLD_RIGHT:
                    right_counter += 1
                else:
                    if right_counter >= MIN_CONSEC_FRAMES and now - last_right_click > CLICK_COOLDOWN:
                        pyautogui.click(button="right")
                        print("Right blink → RIGHT CLICK")
                        last_right_click = now
                    right_counter = 0
        else:
            # No face detected: test mode fallback using the physical mouse
            if TEST_MODE_MOUSE_FALLBACK:
                now = time.time()
                mx, my = pyautogui.position()
                dwell_update_and_maybe_click(mx, my, now)

                if now - _last_progress_print > 0.5:
                    _last_progress_print = now
                    print(f"[TEST MODE: no face] Dwell progress: {_dwell_progress:.2f}")

                time.sleep(TEST_MODE_TICK_SEC)
            else:
                time.sleep(0.05)

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ------------------- UI -------------------
def start_pause():
    if tracking_active.is_set():
        tracking_active.clear()
        toggle_btn.configure(text="Start", image=start_icon)
        status_lbl.configure(text="Status: Paused")
    else:
        tracking_active.set()
        toggle_btn.configure(text="Pause", image=pause_icon)
        status_lbl.configure(text="Status: Running")


def quit_app():
    stop_event.set()
    tracking_active.set()
    root.destroy()

def open_settings():
    global isSettingsOpen
    if isSettingsOpen:
        return
    settings_btn.configure(state="disabled")
    win = ctk.CTkToplevel()
    win.title("Settings")
    win.geometry("360x460")
    win.attributes("-topmost", True)
    isSettingsOpen = True

    def on_close():
        global isSettingsOpen
        isSettingsOpen = False
        settings_btn.configure(state="normal")
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    raw_cameras = utilities.get_available_cameras()

    # 3. Create unique display names
    camera_map = {}
    for cam in raw_cameras:
        display_name = f"{cam['name']}" 
        camera_map[display_name] = cam

    display_names = list(camera_map.keys())

    def on_camera_select(selected_display_name):
        cam_data = camera_map.get(selected_display_name)
        if cam_data:
            utilities.set_camera_input(cam_data["index"])

    dropdown_frame = ctk.CTkFrame(win)
    dropdown_frame.pack(fill="x", padx=10, pady=(10,5))
    
    ctk.CTkLabel(dropdown_frame, text="Select Camera").pack(anchor="w", padx=5)

    camera_menu = ctk.CTkOptionMenu(
        dropdown_frame, 
        values=display_names,
        command=on_camera_select
    )
    camera_menu.pack(fill="x", padx=5, pady=10)

    # 4. Set initial selection based on saved index
    saved_index = utilities.get_camera_input()
    
    current_selection = None
    for name, cam_data in camera_map.items():
        # Match purely on index
        if str(cam_data["index"]) == str(saved_index):
            current_selection = name
            break
    
    if current_selection:
        camera_menu.set(current_selection)
    elif display_names:
        camera_menu.set(display_names[0])

    # Dark/Light mode toggle
    def toggle_mode(choice):
        ctk.set_appearance_mode(choice.lower())

    mode_frame = ctk.CTkFrame(win)
    mode_frame.pack(fill="x", padx=10, pady=(5,0))
    ctk.CTkLabel(mode_frame, text="Appearance").pack(anchor="w", padx=5, pady=5)
    ctk.CTkOptionMenu(mode_frame, values=["Dark", "Light"], command=toggle_mode).pack(fill="x", padx=5, pady=5)

    # Left blink
    left_frame = ctk.CTkFrame(win)
    left_frame.pack(fill="x", padx=10, pady=(10, 5))
    left_frame.columnconfigure(0, weight=1)
    ctk.CTkLabel(left_frame, text="Left Blink Threshold").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    left_val_lbl = ctk.CTkLabel(left_frame, text=f"{EAR_THRESHOLD_LEFT:.2f}")
    left_val_lbl.grid(row=0, column=1, sticky="e", padx=5, pady=5)

    def update_left(v):
        global EAR_THRESHOLD_LEFT
        EAR_THRESHOLD_LEFT = float(v)
        left_val_lbl.configure(text=f"{float(v):.2f}")

    left_slider = ctk.CTkSlider(left_frame, from_=0.1, to=0.5, number_of_steps=50, command=update_left) # pyright: ignore[reportArgumentType]
    left_slider.set(EAR_THRESHOLD_LEFT)
    left_slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    # Right blink
    right_frame = ctk.CTkFrame(win)
    right_frame.pack(fill="x", padx=10, pady=5)
    right_frame.columnconfigure(0, weight=1)
    ctk.CTkLabel(right_frame, text="Right Blink Threshold").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    right_val_lbl = ctk.CTkLabel(right_frame, text=f"{EAR_THRESHOLD_RIGHT:.2f}")
    right_val_lbl.grid(row=0, column=1, sticky="e", padx=5, pady=5)

    def update_right(v):
        global EAR_THRESHOLD_RIGHT
        EAR_THRESHOLD_RIGHT = float(v)
        right_val_lbl.configure(text=f"{float(v):.2f}")

    right_slider = ctk.CTkSlider(right_frame, from_=0.1, to=0.5, number_of_steps=50, command=update_right) # pyright: ignore[reportArgumentType]
    right_slider.set(EAR_THRESHOLD_RIGHT)
    right_slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    # Movement gain
    move_frame = ctk.CTkFrame(win)
    move_frame.pack(fill="x", padx=10, pady=5)
    move_frame.columnconfigure(0, weight=1)
    ctk.CTkLabel(move_frame, text="Head Movement Sensitivity").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    move_val_lbl = ctk.CTkLabel(move_frame, text=f"{MOVEMENT_GAIN:.2f}")
    move_val_lbl.grid(row=0, column=1, sticky="e", padx=5, pady=5)

    def update_gain(v):
        global MOVEMENT_GAIN
        MOVEMENT_GAIN = float(v)
        move_val_lbl.configure(text=f"{float(v):.2f}")

    move_slider = ctk.CTkSlider(move_frame, from_=0.3, to=1.5, number_of_steps=60, command=update_gain) # pyright: ignore[reportArgumentType]
    move_slider.set(MOVEMENT_GAIN)
    move_slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    # Import/Export Settings
    def import_settings():
        fd.askopenfilename(title="Import Settings", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])

    def export_settings():
        fd.asksaveasfilename(title="Export Settings", defaultextension=".json", filetypes=[("JSON files", "*.json")])

    io_frame = ctk.CTkFrame(win)
    io_frame.pack(fill="x", padx=10, pady=5)
    ctk.CTkButton(io_frame, text="Import Settings", command=import_settings).pack(side="left", expand=True, fill="x", padx=5, pady=10)
    ctk.CTkButton(io_frame, text="Export Settings", command=export_settings).pack(side="right", expand=True, fill="x", padx=5, pady=10)


# ------------------- MAIN -------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


root = ctk.CTk()
root.title("EyeOS Control")
root.geometry("800x70")
root.resizable(False, False)

# Dwell progress bar overlay (runs in parallel via Tk's event loop; no blocking)
init_dwell_bar_overlay(root)
root.after(DWELL_BAR_UPDATE_MS, update_dwell_bar_overlay)

bar = ctk.CTkFrame(root)
bar.pack(fill="both", expand=True, padx=8, pady=8)

# Load icons
def load_icon(name, size=(20, 20)):
    return ctk.CTkImage(Image.open(os.path.join("resources", name)), size=size)

start_icon = load_icon("start.png")
pause_icon = load_icon("pause.png")
settings_icon = load_icon("settings.png")
quit_icon = load_icon("quit.png")
voice_icon = load_icon("voice.png")
keyboard_icon = load_icon("keyboard.png")

toggle_btn = ctk.CTkButton(bar, text="Start", image=start_icon, width=100, command=start_pause, compound="left", font=("Arial", 13))
toggle_btn.pack(side="left", padx=4)

status_lbl = ctk.CTkLabel(bar, text="Status: Idle")
status_lbl.pack(side="left", padx=10)

voice_btn = ctk.CTkButton(bar, text="Voice", image=voice_icon, command=lambda: print("Voice Pressed"), compound="left", font=("Arial", 13))
voice_btn.pack(side="left", padx=4)

keyboard_btn = ctk.CTkButton(bar, text="Keyboard", image=keyboard_icon, command=lambda: print("Keyboard Pressed"), compound="left", font=("Arial", 13))
keyboard_btn.pack(side="left", padx=4)

settings_btn = ctk.CTkButton(bar, text="Settings", image=settings_icon, command=open_settings, compound="left", font=("Arial", 13))
settings_btn.pack(side="left", padx=4)

quit_btn = ctk.CTkButton(bar, text="Quit", image=quit_icon, fg_color="#9b1c1c", command=quit_app, compound="left", font=("Arial", 13))
quit_btn.pack(side="right", padx=4)

# Start tracking thread
threading.Thread(target=tracking_loop, daemon=True).start()

root.bind("<space>", lambda e: start_pause())
root.bind("<Escape>", lambda e: quit_app())
root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()