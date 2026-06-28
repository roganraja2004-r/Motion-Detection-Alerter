#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║          Motion Detection Alerter  (single file)         ║
║                                                          ║
║  Run:  python motion_detection_alerter.py                ║
║        python motion_detection_alerter.py --preview      ║
║        python motion_detection_alerter.py --source 1     ║
╚══════════════════════════════════════════════════════════╝

Dependencies:
    pip install opencv-python numpy
    pip install plyer          # optional – desktop notifications
    pip install pygame         # optional – sound alerts

Email alerts: fill in the EMAIL CONFIG section below.
"""

import argparse
import logging
import smtplib
import sys
import threading
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIG  –  edit these values instead of using a YAML file
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Detector ──────────────────────────────────────────────────────────────
    "source": 0,                    # 0 = default webcam | 1 | "/path/to/video.mp4"
    "sensitivity": 500,             # min contour area (px²) to count as motion
    "threshold": 25,                # pixel-difference threshold (0–255)
    "cooldown": 3.0,                # seconds between consecutive alerts
    "save_snapshots": True,         # save annotated JPEG on each alert
    "snapshot_dir": "snapshots",    # folder for saved snapshots
    "show_preview": False,          # open a live preview window

    # ── Alerts ────────────────────────────────────────────────────────────────
    "alert_desktop": True,          # desktop pop-up (requires plyer)
    "alert_sound": False,           # sound alert (requires pygame)
    "sound_file": "alert.wav",      # path to WAV/MP3 file

    # ── Email (set enabled=True and fill in credentials) ──────────────────────
    "email_enabled": False,
    "email_smtp_host": "smtp.gmail.com",
    "email_smtp_port": 587,
    "email_use_tls": True,
    "email_sender": "your_email@gmail.com",
    "email_password": "your_app_password",   # Gmail: use an App Password
    "email_recipients": ["recipient@example.com"],
    "email_subject_prefix": "[Motion Alert]",

    # ── Logging ───────────────────────────────────────────────────────────────
    "log_level": "INFO",
    "log_file": "motion.log",
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO", log_file: str = "motion.log") -> None:
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# ALERTERS
# ═══════════════════════════════════════════════════════════════════════════════

class EmailAlerter:
    """Sends an email with a snapshot attachment via SMTP."""

    def __init__(self, smtp_host, smtp_port, sender, password,
                 recipients, use_tls=True, subject_prefix="[Motion Alert]"):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipients = recipients
        self.use_tls = use_tls
        self.subject_prefix = subject_prefix

    def send(self, frame: np.ndarray, timestamp: datetime) -> None:
        threading.Thread(
            target=self._send_blocking, args=(frame.copy(), timestamp), daemon=True
        ).start()

    def _send_blocking(self, frame: np.ndarray, timestamp: datetime) -> None:
        subject = f"{self.subject_prefix} Motion at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        body = (
            f"Motion detected at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}.\n"
            "A snapshot is attached."
        )
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.attach(MIMEText(body, "plain"))

        _, buf = cv2.imencode(".jpg", frame)
        msg.attach(MIMEImage(buf.tobytes(), name="snapshot.jpg"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as s:
                if self.use_tls:
                    s.starttls()
                s.login(self.sender, self.password)
                s.sendmail(self.sender, self.recipients, msg.as_string())
            logger.info("Email alert sent to %s", self.recipients)
        except Exception as exc:
            logger.error("Email alert failed: %s", exc)


class DesktopAlerter:
    """Shows a system desktop notification (requires plyer)."""

    def send(self, frame: np.ndarray, timestamp: datetime) -> None:
        try:
            from plyer import notification  # type: ignore
            notification.notify(
                title="Motion Detected",
                message=f"Motion at {timestamp.strftime('%H:%M:%S')}",
                timeout=5,
            )
        except ImportError:
            logger.warning("plyer not installed – desktop alerts disabled.")
        except Exception as exc:
            logger.error("Desktop alert failed: %s", exc)


class SoundAlerter:
    """Plays a sound file (requires pygame)."""

    def __init__(self, sound_file: str = "alert.wav"):
        self.sound_file = Path(sound_file)
        try:
            import pygame  # type: ignore
            pygame.mixer.init()
            self._ok = True
        except ImportError:
            logger.warning("pygame not installed – sound alerts disabled.")
            self._ok = False

    def send(self, frame: np.ndarray, timestamp: datetime) -> None:
        if not self._ok:
            return
        try:
            import pygame  # type: ignore
            if self.sound_file.exists():
                pygame.mixer.Sound(str(self.sound_file)).play()
            else:
                logger.warning("Sound file not found: %s", self.sound_file)
        except Exception as exc:
            logger.error("Sound alert failed: %s", exc)


class AlertManager:
    """Combines multiple alerters into one callable."""

    def __init__(self):
        self._alerters = []

    def add(self, alerter) -> None:
        self._alerters.append(alerter)

    def notify(self, frame: np.ndarray, timestamp: datetime) -> None:
        for alerter in self._alerters:
            try:
                alerter.send(frame, timestamp)
            except Exception as exc:
                logger.error("Alerter %s error: %s", type(alerter).__name__, exc)

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MotionDetector:
    """
    Detects motion from a video source using frame differencing.

    Parameters
    ----------
    source        : Camera index (int) or video file path (str)
    sensitivity   : Minimum contour area in px² to count as motion
    threshold     : Pixel-difference threshold for binarization (0–255)
    cooldown      : Seconds to wait between consecutive alerts
    save_snapshots: Save annotated JPEG on each alert
    snapshot_dir  : Directory to store snapshots
    on_motion     : Callback(frame, datetime) invoked on detected motion
    """

    def __init__(
        self,
        source: int | str = 0,
        sensitivity: int = 500,
        threshold: int = 25,
        cooldown: float = 3.0,
        save_snapshots: bool = True,
        snapshot_dir: str = "snapshots",
        on_motion: Optional[Callable[[np.ndarray, datetime], None]] = None,
    ):
        self.source = source
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.cooldown = cooldown
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir)
        self.on_motion = on_motion

        self._cap: Optional[cv2.VideoCapture] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._last_alert_time: float = 0.0
        self._running = False

        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self, show_preview: bool = False) -> None:
        """Open the video source and begin the detection loop."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        logger.info("Motion detection started  (source=%s)", self.source)
        self._running = True
        try:
            self._loop(show_preview)
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Motion detection stopped.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self, show_preview: bool) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame – source may have ended.")
                break

            motion, contours = self._analyse(frame)

            if motion and (time.time() - self._last_alert_time) >= self.cooldown:
                self._last_alert_time = time.time()
                now = datetime.now()
                logger.info("Motion detected at %s", now.strftime("%Y-%m-%d %H:%M:%S"))
                self._handle_motion(frame, now, contours)

            if show_preview:
                preview = self._draw_contours(frame.copy(), contours)
                cv2.imshow("Motion Detector  –  press Q to quit", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def _analyse(self, frame: np.ndarray) -> tuple[bool, list]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_frame is None:
            self._prev_frame = gray
            return False, []

        delta = cv2.absdiff(self._prev_frame, gray)
        self._prev_frame = gray

        _, thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) >= self.sensitivity]
        return bool(significant), significant

    def _handle_motion(self, frame: np.ndarray, timestamp: datetime, contours: list) -> None:
        if self.save_snapshots:
            filename = self.snapshot_dir / f"motion_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), self._draw_contours(frame.copy(), contours))
            logger.info("Snapshot saved: %s", filename)

        if self.on_motion:
            self.on_motion(frame, timestamp)

    @staticmethod
    def _draw_contours(frame: np.ndarray, contours: list) -> np.ndarray:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"MOTION  {datetime.now().strftime('%H:%M:%S')}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return frame

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motion Detection Alerter")
    parser.add_argument("--source", default=None,
                        help="Camera index (0,1,…) or video file path")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--sensitivity", type=int, default=None,
                        help="Min contour area in px² (default: 500)")
    parser.add_argument("--cooldown", type=float, default=None,
                        help="Seconds between alerts (default: 3.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # CLI overrides
    if args.source is not None:
        CONFIG["source"] = int(args.source) if str(args.source).isdigit() else args.source
    if args.sensitivity is not None:
        CONFIG["sensitivity"] = args.sensitivity
    if args.cooldown is not None:
        CONFIG["cooldown"] = args.cooldown

    setup_logging(CONFIG["log_level"], CONFIG["log_file"])

    # Build alert manager
    manager = AlertManager()
    if CONFIG["alert_desktop"]:
        manager.add(DesktopAlerter())
    if CONFIG["alert_sound"]:
        manager.add(SoundAlerter(CONFIG["sound_file"]))
    if CONFIG["email_enabled"]:
        manager.add(EmailAlerter(
            smtp_host=CONFIG["email_smtp_host"],
            smtp_port=CONFIG["email_smtp_port"],
            sender=CONFIG["email_sender"],
            password=CONFIG["email_password"],
            recipients=CONFIG["email_recipients"],
            use_tls=CONFIG["email_use_tls"],
            subject_prefix=CONFIG["email_subject_prefix"],
        ))

    detector = MotionDetector(
        source=CONFIG["source"],
        sensitivity=CONFIG["sensitivity"],
        threshold=CONFIG["threshold"],
        cooldown=CONFIG["cooldown"],
        save_snapshots=CONFIG["save_snapshots"],
        snapshot_dir=CONFIG["snapshot_dir"],
        on_motion=manager.notify,
    )

    print("=" * 52)
    print("  Motion Detection Alerter")
    print(f"  Source      : {CONFIG['source']}")
    print(f"  Sensitivity : {CONFIG['sensitivity']} px²")
    print(f"  Cooldown    : {CONFIG['cooldown']} s")
    print(f"  Snapshots   : {CONFIG['snapshot_dir']}/")
    print(f"  Preview     : {args.preview or CONFIG['show_preview']}")
    print("  Press Ctrl-C or Q (preview) to stop.")
    print("=" * 52)

    try:
        detector.start(show_preview=args.preview or CONFIG["show_preview"])
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
