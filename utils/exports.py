"""Bounded recording exports, independent of detector/model initialization."""
from contextlib import contextmanager, suppress
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def find_ffmpeg():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    common_paths = [
        '/opt/homebrew/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/usr/bin/ffmpeg'
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return 'ffmpeg'


def recording_directory(base_dir, camera, date):
    for value in (camera, date):
        if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in ("/", "\\", "\0")):
            raise ValueError("Camera and date must be single directory names")
    root = (Path(base_dir) / "cameras").resolve()
    stream = (root / camera / "streams" / date).resolve()
    try:
        stream.relative_to(root)
    except ValueError as exc:
        raise ValueError("Recording path is outside camera storage") from exc
    if not stream.is_dir():
        raise FileNotFoundError("Recording directory does not exist")
    return stream


@contextmanager
def exported_recording(base_dir, camera, date):
    stream = recording_directory(base_dir, camera, date)
    init = stream / "init.mp4"
    segments = sorted(stream.glob("stream_*.m4s"))
    if not init.is_file() or not segments:
        raise FileNotFoundError("Recording is incomplete")
    for path in [init, *segments]:
        if path.is_symlink():
            raise ValueError("Recording files must not be symbolic links")
    ffmpeg = find_ffmpeg()
    if ffmpeg == "ffmpeg" and not shutil.which(ffmpeg):
        raise RuntimeError("FFmpeg is not available")
    # Every request owns its output; concurrent exports cannot overwrite it.
    with tempfile.TemporaryDirectory(prefix=".export-", dir=stream) as temporary:
        output = Path(temporary) / "recording.mp4"
        proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-i", "pipe:0",
                                 "-c", "copy", "-movflags", "+faststart", str(output)],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            for path in [init, *segments]:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, proc.stdin)
            proc.stdin.close()
            if proc.wait() != 0 or not output.is_file():
                raise RuntimeError("FFmpeg export failed")
            yield output
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg rejected recording input") from exc
        finally:
            if proc.stdin and not proc.stdin.closed:
                with suppress(BrokenPipeError):
                    proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
