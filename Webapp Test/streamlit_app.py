# Streamlit version of the MediaPipe posture monitor.
# Ref: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/
import base64
import hashlib
import io
import math as m
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Posture Monitor", page_icon="🧍", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, GREEN, LGREEN = (50, 50, 255), (127, 255, 0), (127, 233, 100)
YELLOW, PINK, WHITE = (0, 255, 255), (255, 0, 255), (255, 255, 255)
OUT_DIR = os.path.join(BASE, "outputs")
recorder = components.declare_component("sound_recorder", path=os.path.join(BASE, "sound_recorder"))
STREAMLIT_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
MAX_W = 960  # uploaded videos are downscaled to this width for speed


def sound_path():
    """Use Fix_Your_Posture.wav if it sits next to this file, else make a beep."""
    p = os.path.join(BASE, "Fix_Your_Posture.wav")
    if os.path.exists(p):
        return p
    p = os.path.join(tempfile.gettempdir(), "posture_beep.wav")
    if not os.path.exists(p):
        sr = 22050
        t = np.linspace(0, 0.6, int(sr * 0.6), False)
        tone = (np.sin(2 * np.pi * 880 * t) * 0.4 * 32767).astype(np.int16)
        with wave.open(p, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(tone.tobytes())
    return p


SOUND = sound_path()
with open(SOUND, "rb") as _f:
    SOUND_BYTES = _f.read()


def active_sound():
    """(bytes, file_path, mime) of the user's own sound if there is one, else the default alert.
    Anything that isn't a plain WAV is converted with the ffmpeg that ships with moviepy."""
    import subprocess

    c = st.session_state.get("custom_sound")
    if not c:
        return SOUND_BYTES, SOUND, "audio/wav"
    tag = hashlib.md5(c["data"]).hexdigest()[:12]
    wav = os.path.join(tempfile.gettempdir(), f"posture_custom_{tag}.wav")
    if not os.path.exists(wav):
        try:
            src = os.path.join(tempfile.gettempdir(), f"posture_custom_{tag}.{c['ext']}")
            with open(src, "wb") as f:
                f.write(c["data"])
            if c["ext"] == "wav":
                os.replace(src, wav)
            else:
                import imageio_ffmpeg
                r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src, "-ar", "44100", "-ac", "1", wav],
                                   capture_output=True, timeout=60)
                if r.returncode != 0 or not os.path.exists(wav):
                    raise RuntimeError(r.stderr.decode(errors="ignore")[-200:])
        except Exception as e:
            st.sidebar.warning(f"Couldn't read your sound ({e}); using the default.")
            return SOUND_BYTES, SOUND, "audio/wav"
    with open(wav, "rb") as f:
        return f.read(), wav, "audio/wav"


def find_distance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_angle(x1, y1, x2, y2):
    """Angle of the line (x1,y1)->(x2,y2) relative to the vertical, in degrees."""
    return -1 * m.degrees(m.atan2(x2 - x1, y1 - y2))


@dataclass
class Params:
    offset: int = 100
    neck: int = 25
    torso: int = 10
    hold: float = 3.0
    cooldown: float = 5.0


class Tracker:
    """Per-session posture state. Same maths and overlay as the original script."""

    def __init__(self, complexity=1, skip=1):
        self._pose = None
        self.complexity, self.skip, self.n, self.last_lm = complexity, max(1, skip), 0, None
        self.good = self.bad = self.streak = 0.0
        self.last_warn = -1e9
        self.t, self.neck, self.torso, self.warnings = [], [], [], []

    @property
    def pose(self):
        if self._pose is None:  # created lazily so it lives in the thread that uses it
            self._pose = mp.solutions.pose.Pose(model_complexity=self.complexity)
        return self._pose

    def process(self, img, t, dt, p):
        """Annotate a BGR frame. t = seconds into the session/video, dt = frame step.
        Returns (annotated_frame, warning_triggered)."""
        h, w = img.shape[:2]
        self.n += 1
        if self.skip > 1 and self.n % self.skip and self.last_lm is not None:
            lm = self.last_lm  # skipped frame: reuse the previous detection
        else:
            lm = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).pose_landmarks
            self.last_lm = lm
        if lm is None:
            cv2.putText(img, "No person detected", (10, 30), FONT, 0.9, RED, 2)
            return img, False

        P = mp.solutions.pose.PoseLandmark
        pt = lambda k: (int(lm.landmark[k].x * w), int(lm.landmark[k].y * h))
        (lsx, lsy), (rsx, rsy) = pt(P.LEFT_SHOULDER), pt(P.RIGHT_SHOULDER)
        lex, ley = pt(P.LEFT_EAR)
        lhx, lhy = pt(P.LEFT_HIP)

        offset = find_distance(lsx, lsy, rsx, rsy)
        if offset < p.offset:
            cv2.putText(img, f"{int(offset)} Shoulders aligned", (w - 280, 30), FONT, 0.6, GREEN, 2)
        else:
            cv2.putText(img, f"{int(offset)} Shoulders not aligned", (w - 280, 30), FONT, 0.6, RED, 2)

        neck = find_angle(lsx, lsy, lex, ley)
        torso = find_angle(lhx, lhy, lsx, lsy)
        self.t.append(t)
        self.neck.append(neck)
        self.torso.append(torso)

        for c, col, th in [((lsx, lsy), WHITE, 2), ((lex, ley), WHITE, 2), ((lsx, lsy - 100), WHITE, 2),
                           ((rsx, rsy), PINK, -1), ((lhx, lhy), YELLOW, -1), ((lhx, lhy - 100), YELLOW, -1)]:
            cv2.circle(img, c, 7, col, th)

        warned = False
        if abs(neck) < p.neck:
            self.streak = 0.0
            self.good += dt
            txt, line = LGREEN, GREEN
        else:
            self.bad += dt
            self.streak += dt
            txt, line = RED, RED
            if self.streak >= p.hold and t - self.last_warn >= p.cooldown:
                self.warnings.append(t)
                self.last_warn = t
                warned = True

        cv2.putText(img, f"Neck inclination: {int(neck)}", (10, 30), FONT, 0.6, txt, 2)
        cv2.putText(img, f"Torso inclination: {int(torso)}", (10, 60), FONT, 0.6, txt, 2)
        cv2.putText(img, str(int(neck)), (lsx + 10, lsy), FONT, 0.9, txt, 2)
        cv2.putText(img, str(int(torso)), (lhx + 10, lhy), FONT, 0.9, txt, 2)
        cv2.line(img, (lsx, lsy), (lex, ley), line, 2)
        cv2.line(img, (lsx, lsy), (lsx, lsy - 100), line, 2)
        cv2.line(img, (lhx, lhy), (lsx, lsy), line, 2)
        cv2.line(img, (lhx, lhy), (lhx, lhy - 100), line, 2)
        cv2.putText(img, f"Good Posture Time : {round(self.good, 1)}s", (10, h - 50), FONT, 0.9, GREEN, 2)
        cv2.putText(img, f"Bad Posture Time : {round(self.bad, 1)}s", (10, h - 20), FONT, 0.9, RED, 2)
        return img, warned

    def snapshot(self):
        return dict(t=list(self.t), neck=list(self.neck), torso=list(self.torso),
                    good=self.good, bad=self.bad, warnings=list(self.warnings))


def show_figure(fig):
    """Embed the chart as an inline PNG (avoids Streamlit's media endpoint)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%">', unsafe_allow_html=True)


def show_video(data):
    """Embed the video inline. Use the player's ⋮ menu -> Download to save it."""
    if len(data) > 40e6:
        st.info("Video is too large to preview here - open the saved file instead.")
        return
    b64 = base64.b64encode(data).decode()
    components.html(
        f'<video controls playsinline style="width:100%;max-height:480px;background:#000" '
        f'src="data:video/mp4;base64,{b64}"></video>', height=490)


def make_plot(s, p):
    fig, ax = plt.subplots(figsize=(12, 5))
    t, neck, torso = s["t"], np.array(s["neck"]), np.array(s["torso"])
    ax.plot(t, neck, color="tab:blue", label="Neck Inclination")
    ax.plot(t, torso, color="tab:orange", label="Torso Inclination")
    ax.plot(t, np.where(np.abs(neck) > p.neck, neck, np.nan), color="tab:red", zorder=3, label="Out of Tolerance")
    ax.plot(t, np.where(np.abs(torso) > p.torso, torso, np.nan), color="tab:red", zorder=3)
    for v, c in [(p.neck, "tab:blue"), (p.torso, "tab:orange")]:
        ax.axhline(v, color=c, linestyle="--")
        ax.axhline(-v, color=c, linestyle="--")
    ax.set(xlabel="Time (s)", ylabel="Inclination Angle (degrees)", title="Posture Inclination Over Time")
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig


def show_results(s, p, key):
    tot = s["good"] + s["bad"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Good posture", f"{s['good']:.1f}s")
    c2.metric("Bad posture", f"{s['bad']:.1f}s")
    c3.metric("% good", f"{(100 * s['good'] / tot) if tot else 0:.0f}%")
    c4.metric("Warnings", len(s["warnings"]))
    if s["t"]:
        fig = make_plot(s, p)
        show_figure(fig)
        plt.close(fig)


# ---------------- Live webcam (runs in the visitor's browser) ----------------
REC_FPS = 20


def live_tab(p, sound_on):
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            # Lighter pose model + detect every 2nd frame keeps the stream real-time.
            self.tracker, self.params = Tracker(complexity=0, skip=2), Params()
            self.lock = threading.Lock()
            self.t0 = self.prev = None
            self.warn_count = 0
            self.raw_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            self.writer, self.written, self.closed = None, 0, False

        def _record(self, img, t):
            if self.writer is None:
                h, w = img.shape[:2]
                self.writer = cv2.VideoWriter(self.raw_path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (w, h))
            # Constant-fps recording: repeat the frame to fill gaps so video time == real time.
            for _ in range(max(0, min(int(t * REC_FPS) + 1 - self.written, 60))):
                self.writer.write(img)
                self.written += 1

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            now = time.time()
            if self.t0 is None:
                self.t0 = self.prev = now
            dt, self.prev = min(now - self.prev, 0.5), now
            with self.lock:
                out, warned = self.tracker.process(img, now - self.t0, dt, self.params)
                self.warn_count += warned
                if not self.closed:
                    self._record(out, now - self.t0)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

        def stats(self):
            with self.lock:
                tr = self.tracker
                return dict(good=tr.good, bad=tr.bad, warn_count=self.warn_count)

        def snapshot(self):
            with self.lock:
                return self.tracker.snapshot()

        def finish(self):
            with self.lock:
                self.closed = True
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None
                return self.raw_path if self.written else None

    ctx = webrtc_streamer(
        key="posture-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 20}},
            "audio": False,
        },
        video_processor_factory=LiveProcessor,
        async_processing=True,
    )
    st.caption("Sit side-on to the camera so your left side faces it. Press START and allow camera access. "
               "The annotated recording is saved when you press STOP.")
    stats, alert, seen = st.empty(), st.empty(), 0
    snd_bytes, _, snd_mime = active_sound()
    sound_b64 = base64.b64encode(snd_bytes).decode()
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc:
            proc.params = p
            st.session_state["live_proc"] = proc
            snap = proc.stats()
            with stats.container():
                a, b, c = st.columns(3)
                a.metric("Good posture", f"{snap['good']:.0f}s")
                b.metric("Bad posture", f"{snap['bad']:.0f}s")
                c.metric("Warnings", snap["warn_count"])
            if snap["warn_count"] > seen:
                seen = snap["warn_count"]
                st.toast("Posture violation! Sit straight up!", icon="⚠️")
                if sound_on:
                    alert.empty()
                    with alert.container():
                        # Plain HTML audio: works on every Streamlit version, and the
                        # changing comment makes each alert a fresh element that autoplays.
                        components.html(
                            f'<audio autoplay src="data:{snd_mime};base64,{sound_b64}"></audio><!--{seen}-->',
                            height=0)
        time.sleep(1.0)

    if not ctx.state.playing:
        proc = st.session_state.pop("live_proc", None)
        if proc is not None:  # a session just ended: finish the recording and keep the results
            raw, snap = proc.finish(), proc.snapshot()
            video = saved = None
            if raw:
                with st.spinner("Saving recording…"):
                    try:
                        video = encode_with_audio(raw, snap["warnings"], active_sound()[1])
                        os.makedirs(OUT_DIR, exist_ok=True)
                        saved = os.path.join(OUT_DIR, time.strftime("live_%Y%m%d_%H%M%S.mp4"))
                        with open(saved, "wb") as f:
                            f.write(video)
                    except Exception as e:
                        st.error(f"Could not save the recording: {e}")
            st.session_state["live_last"] = dict(snap=snap, video=video, saved=saved, params=p)
        r = st.session_state.get("live_last")
        if r:
            st.subheader("Last live session")
            if r["video"]:
                show_video(r["video"])
                st.caption(f"{len(r['video']) / 1e6:.1f} MB · saved to: {r['saved']}")
            show_results(r["snap"], r["params"], "live")


# ---------------- Uploaded video ----------------
def analyze_video(path, p, bar):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open that video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 0 < fps <= 240 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = min(1.0, MAX_W / w0)
    w, h = int(w0 * k) // 2 * 2, int(h0 * k) // 2 * 2
    raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tr, i = Tracker(), 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out, _ = tr.process(cv2.resize(img, (w, h)), i / fps, 1 / fps, p)
        writer.write(out)
        i += 1
        if i % 5 == 0:
            bar.progress(min(i / total, 1.0), text=f"Analyzing frame {i} of ~{total}")
    cap.release()
    writer.release()
    bar.progress(1.0, text="Encoding video…")
    return raw, tr.snapshot()


def encode_with_audio(raw, warnings, sound_file=None):
    """Re-encode to browser-friendly H.264 and lay the warning sound over the video."""
    from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    clip, clips, mix, final = VideoFileClip(raw), [], None, None
    try:
        final = clip
        for t in warnings:
            if 0 <= t < clip.duration:
                clips.append(AudioFileClip(sound_file or SOUND).with_start(t))
        if clips:
            mix = CompositeAudioClip(clips).with_duration(clip.duration)
            final = clip.with_audio(mix)
        final.write_videofile(out, codec="libx264", audio_codec="aac", preset="ultrafast",
                              ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
    finally:
        for c in clips + [mix, clip]:
            if c is not None:
                c.close()
    with open(out, "rb") as f:
        return f.read()


def video_tab(p):
    up = st.file_uploader("Upload a side-view video", type=["mp4", "mov", "avi", "mkv"])
    if up and st.button("Analyze video", type="primary"):
        src = tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False)
        src.write(up.getbuffer())
        src.close()
        bar = st.progress(0.0, text="Starting…")
        try:
            raw, snap = analyze_video(src.name, p, bar)
            data = encode_with_audio(raw, snap["warnings"], active_sound()[1])
            os.makedirs(OUT_DIR, exist_ok=True)
            saved = os.path.join(OUT_DIR, "posture_overlay_with_audio.mp4")
            with open(saved, "wb") as f:
                f.write(data)
            st.session_state["vid"] = dict(snap=snap, video=data, params=p, saved=saved)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        bar.empty()
    r = st.session_state.get("vid")
    if r:
        show_video(r["video"])
        st.caption(f"{len(r['video']) / 1e6:.1f} MB · saved to: {r['saved']}")
        show_results(r["snap"], r["params"], "vid")
        if r["snap"]["warnings"]:
            st.caption("Warning timestamps (s): " + ", ".join(f"{t:.1f}" for t in r["snap"]["warnings"]))


# ---------------- Page ----------------
st.title("🧍 Posture Monitor")
with st.sidebar:
    st.header("Settings")
    params = Params(
        offset=st.slider("Shoulder offset threshold (px)", 10, 400, 100,
                         help="Pixels in the processed frame; larger means 'not aligned' triggers later."),
        neck=st.slider("Neck angle threshold (°)", 5, 60, 25),
        torso=st.slider("Torso angle threshold (°)", 1, 45, 10,
                        help="Shown on the chart only, as in the original script."),
        hold=st.slider("Bad posture before alert (s)", 1.0, 10.0, 3.0, 0.5),
        cooldown=st.slider("Minimum gap between alerts (s)", 1.0, 30.0, 5.0, 1.0),
    )
    sound_on = st.checkbox("Play alert sound (live)", value=True)
    st.subheader("🎤 Your own warning sound")
    st.caption("Record your own alert (a few seconds is plenty)")
    with st.container():
        rec = recorder(key="sound_rec", default=None)
    up_snd = st.file_uploader("…or upload a sound file", type=["wav", "mp3", "m4a", "ogg"])
    custom = None
    if rec:
        custom = dict(data=base64.b64decode(rec["data"]), ext=rec["ext"], mime=rec["mime"])
    elif up_snd is not None:
        ext = os.path.splitext(up_snd.name)[1].lstrip(".").lower() or "wav"
        custom = dict(data=up_snd.getvalue(), ext=ext, mime=up_snd.type or f"audio/{ext}")
    st.session_state["custom_sound"] = custom
    st.caption("Your sound replaces the default alert, live and in analyzed videos. "
               "Clear it with the ✕ to go back to the default.")

tab1, tab2 = st.tabs(["📷 Live webcam", "🎞️ Upload video"])
with tab1:
    live_tab(params, sound_on)
with tab2:
    video_tab(params)
