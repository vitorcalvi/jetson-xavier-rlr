"""Multiprocess, thermal-aware RLR prototype for Jetson Xavier NX."""
from __future__ import annotations

import argparse
import errno
import fcntl
import glob
import json
import multiprocessing as mp
import os
import queue
import signal
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror, request

from logic import build_prompt, dedupe_detections, vad_chunks

COCO = "person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed dining table toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink refrigerator book clock vase scissors teddy bear hair drier toothbrush".split()
_ENGINE_CACHE = {}
_ACTIVE_CHILDREN = []
_ACTIVE_STOP = None


def _acquire_run_lock(path="/models/proto/run.lock"):
    """Hold an exclusive, non-blocking process lock for the full run."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise RuntimeError("another prototype run holds " + path) from exc
        raise
    return fd


def _wait_for_health(base_url, timeout=180, interval=1, get=request.urlopen, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    last_error = "no response"
    while clock() < deadline:
        try:
            with get(base_url.rstrip("/") + "/health", timeout=min(5, max(0.1, deadline - clock()))) as response:
                if response.status == 200:
                    return
                last_error = "HTTP %d" % response.status
        except (OSError, urlerror.URLError) as exc:
            last_error = str(exc)
        sleep(min(interval, max(0, deadline - clock())))
    raise RuntimeError("llama-server health timeout after %ds: %s" % (timeout, last_error))


def _temps() -> list[float]:
    out = []
    for path in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        try:
            out.append(int(Path(path).read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return out


def _self_state(detections: list[str], transcript: str) -> str:
    temps = _temps()
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                mem[key] = value.strip()
    except OSError:
        pass
    try:
        power = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        power = "unavailable"
    guard = bool(subprocess.run(["pgrep", "-f", "^/bin/sh /usr/local/bin/thermal-guard.sh"], capture_output=True).returncode == 0)
    return json.dumps({"thermal_zone_c": temps, "ram": mem, "power_mode": power,
                       "thermal_guard_active": guard, "recent_detections": detections,
                       "last_transcript": transcript}, ensure_ascii=False)


def _require_safe_device():
    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=3, check=True).stdout
    if not re.search(r"(?m)^\s*5\s*$", mode):
        raise RuntimeError("safety stop: Jetson must already be in nvpmodel ID 5; observed " + repr(mode.strip()))
    guard = subprocess.run(["pgrep", "-f", "^/bin/sh /usr/local/bin/thermal-guard.sh"], capture_output=True)
    if guard.returncode:
        raise RuntimeError("safety stop: /usr/local/bin/thermal-guard.sh is not running")


def _image(source, size: int):
    import numpy as np
    try:
        from PIL import Image
        im = (Image.open(source).convert("RGB") if isinstance(source, str)
              else Image.fromarray(source).convert("RGB"))
        w, h = im.size
        scale = min(size / w, size / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = im.resize((nw, nh))
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        return np.asarray(canvas, dtype=np.float32) / 255.0, (w, h), scale, ((size - nw) // 2, (size - nh) // 2)
    except ImportError:
        import cv2
        bgr = cv2.imread(source) if isinstance(source, str) else source[:, :, ::-1]
        if bgr is None:
            raise ValueError("cannot decode image " + str(source))
        h, w = bgr.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = cv2.resize(bgr, (nw, nh))[:, :, ::-1]
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        x, y = (size - nw) // 2, (size - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        return canvas.astype(np.float32) / 255.0, (w, h), scale, (x, y)


def _nms(rows, threshold=0.45):
    import numpy as np
    if len(rows) == 0:
        return []
    keep = []
    for cls in set(int(r[5]) for r in rows):
        candidates = [r for r in rows if int(r[5]) == cls]
        candidates.sort(key=lambda r: r[4], reverse=True)
        while candidates:
            best = candidates.pop(0)
            keep.append(best)
            rest = []
            for row in candidates:
                x1, y1 = max(best[0], row[0]), max(best[1], row[1])
                x2, y2 = min(best[2], row[2]), min(best[3], row[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = max(0, best[2] - best[0]) * max(0, best[3] - best[1])
                area_b = max(0, row[2] - row[0]) * max(0, row[3] - row[1])
                if inter / max(area_a + area_b - inter, 1e-6) <= threshold:
                    rest.append(row)
            candidates = rest
    return keep


def _infer(engine_path: str, image_path):
    import ctypes
    import numpy as np
    import tensorrt as trt
    cached = _ENGINE_CACHE.get(engine_path)
    if cached is None:
        cuda = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        context_handle = ctypes.c_void_p()
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cuda.cuCtxSynchronize.argtypes = []
        for call, args in ((cuda.cuInit, (0,)), (cuda.cuDeviceGet, (ctypes.byref(device), 0)),
                           (cuda.cuCtxCreate_v2, (ctypes.byref(context_handle), 0, device.value))):
            result = call(*args)
            if result:
                raise RuntimeError("CUDA driver call failed with code %d" % result)
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the YOLO engine")
        context = engine.create_execution_context()
        allocations = {}
        _ENGINE_CACHE[engine_path] = (runtime, engine, context, cuda, context_handle, allocations)
    else:
        runtime, engine, context, cuda, context_handle, allocations = cached
    if cuda.cuCtxSetCurrent(context_handle):
        raise RuntimeError("could not make the TensorRT CUDA context current")
    bindings = [0] * engine.num_bindings
    input_idx = next(i for i in range(engine.num_bindings) if engine.binding_is_input(i))
    shape = tuple(context.get_binding_shape(input_idx))
    if any(x < 0 for x in shape):
        shape = tuple(engine.get_binding_shape(input_idx))
        context.set_binding_shape(input_idx, shape)
    size = int(shape[-1])
    image, original, scale, pad = _image(image_path, size)
    data = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=trt.nptype(engine.get_binding_dtype(input_idx)))
    host_outputs = []
    for idx in range(engine.num_bindings):
        shp = tuple(context.get_binding_shape(idx))
        if any(x < 0 for x in shp):
            raise RuntimeError("unresolved TensorRT binding shape: " + str(shp))
        dtype = trt.nptype(engine.get_binding_dtype(idx))
        host = np.empty(int(np.prod(shp)), dtype=dtype)
        nbytes = host.nbytes if idx != input_idx else data.nbytes
        if idx not in allocations:
            pointer = ctypes.c_uint64()
            result = cuda.cuMemAlloc_v2(ctypes.byref(pointer), nbytes)
            if result:
                raise RuntimeError("CUDA allocation failed with code %d" % result)
            allocations[idx] = pointer.value
        bindings[idx] = allocations[idx]
        if idx == input_idx:
            result = cuda.cuMemcpyHtoD_v2(allocations[idx], ctypes.c_void_p(data.ctypes.data), data.nbytes)
            if result:
                raise RuntimeError("CUDA input copy failed with code %d" % result)
        else:
            host_outputs.append((idx, host, shp))
    if not context.execute_v2(bindings):
        raise RuntimeError("TensorRT execution failed")
    if cuda.cuCtxSynchronize():
        raise RuntimeError("CUDA synchronization failed")
    for idx, host, _ in host_outputs:
        result = cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(host.ctypes.data), allocations[idx], host.nbytes)
        if result:
            raise RuntimeError("CUDA output copy failed with code %d" % result)
    if not host_outputs:
        raise RuntimeError("YOLO engine has no output binding")
    out = host_outputs[0][1].reshape(host_outputs[0][2])
    out = np.squeeze(out)
    if out.ndim != 2:
        raise RuntimeError("unsupported YOLO output shape " + str(out.shape))
    if out.shape[0] < out.shape[1]:
        out = out.T
    rows = []
    ow, oh = original
    px, py = pad
    for pred in out:
        if len(pred) < 6:
            continue
        scores = pred[4:]
        cls = int(np.argmax(scores))
        confidence = float(scores[cls])
        if confidence < 0.25 or cls >= len(COCO):
            continue
        cx, cy, bw, bh = map(float, pred[:4])
        x1 = max(0, min(ow, (cx - bw / 2 - px) / scale))
        y1 = max(0, min(oh, (cy - bh / 2 - py) / scale))
        x2 = max(0, min(ow, (cx + bw / 2 - px) / scale))
        y2 = max(0, min(oh, (cy + bh / 2 - py) / scale))
        rows.append([x1, y1, x2, y2, confidence, cls])
    detections = ["[V:%s:%.2f]" % (COCO[int(row[5])], row[4]) for row in _nms(rows)[:20]]
    return detections


def _vision_proc(events, stop, engine: str, images: list[str], requested_fps: float, camera_index=None):
    try:
        if not images and camera_index is None:
            raise RuntimeError("no camera or COCO image inputs found")
        capture = None
        if camera_index is not None:
            import cv2
            capture = cv2.VideoCapture(camera_index)
            if not capture.isOpened():
                raise RuntimeError("cannot open camera index %s" % camera_index)
        previous = None
        index = count = 0
        window = time.monotonic()
        last_fps = 0.0
        while not stop.is_set():
            temps = _temps()
            if temps and max(temps) >= 78:
                while not stop.is_set() and max(_temps() or [0]) >= 70:
                    stop.wait(2)
                continue
            fps = 1.0 if temps and max(temps) >= 75 else requested_fps
            if capture is not None:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("camera frame read failed")
                source = frame[:, :, ::-1]
            else:
                source = images[index % len(images)]
                index += 1
            detections = _infer(engine, source)
            detections, changed = dedupe_detections(previous, detections)
            if changed:
                events.put({"type": "vision", "at": time.monotonic(), "detections": detections})
                previous = detections
            count += 1
            now = time.monotonic()
            if now - window >= 1:
                last_fps = count / (now - window)
                events.put({"type": "fps", "fps": last_fps})
                count, window = 0, now
            stop.wait(max(0, 1 / fps - (time.monotonic() - now)))
    except Exception as exc:
        events.put({"type": "error", "source": "vision", "error": str(exc), "trace": traceback.format_exc()})


def _process_entry(target, args):
    os.setsid()
    target(*args)


def _whisper(audio: bytes, model: str, binary: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio)
        proc = subprocess.run([binary, "-m", model, "-f", tmp.name, "-ng", "-t", "4", "--no-fallback"],
                              capture_output=True, text=True, timeout=90, start_new_session=True)
        if proc.returncode:
            raise RuntimeError("whisper exit %d: %s" % (proc.returncode, proc.stderr[-500:]))
        lines = []
        for line in proc.stdout.splitlines():
            line = re.sub(r"^\[[^\]]+\]\s*", "", line).strip()
            if line and not line.startswith("whisper_print_timings:"):
                lines.append(line)
        return " ".join(lines)


def _vad_file(path: str, model: str, binary: str, mode: int, events):
    import audioop
    import webrtcvad
    with wave.open(path, "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
    if channels > 1:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    if rate != 16000:
        raw, _ = audioop.ratecv(raw, 2, 1, rate, 16000, None)
    vad = webrtcvad.Vad(mode)
    chunks = vad_chunks(raw, 16000, 20)
    active, silent = [], 0
    start = None
    for frame in chunks:
        voiced = vad.is_speech(frame, 16000)
        if voiced:
            start = start or time.monotonic()
            active.append(frame)
            silent = 0
        elif active:
            active.append(frame)
            silent += 1
            if silent >= 15:
                pcm = b"".join(active[:-15])
                if pcm:
                    transcript = _whisper(pcm, model, binary)
                    events.put({"type": "transcript", "at": time.monotonic(), "vad_at": start,
                                "transcript": transcript, "source": Path(path).name})
                active, silent, start = [], 0, None
    if active:
        pcm = b"".join(active)
        transcript = _whisper(pcm, model, binary)
        events.put({"type": "transcript", "at": time.monotonic(), "vad_at": start,
                    "transcript": transcript, "source": Path(path).name})


def _audio_proc(events, stop, files: list[str], model: str, binary: str, mode: int):
    try:
        if not files:
            raise RuntimeError("no microphone or WAV input found")
        for path in files:
            if stop.is_set():
                break
            _vad_file(path, model, binary, mode, events)
    except Exception as exc:
        events.put({"type": "error", "source": "audio", "error": str(exc), "trace": traceback.format_exc()})


def _live_audio_proc(events, stop, device: str, model: str, binary: str, mode: int):
    """Capture raw mono PCM from arecord and apply the same 20 ms VAD gate."""
    import webrtcvad
    proc = None
    try:
        proc = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", "16000",
                                 "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        vad = webrtcvad.Vad(mode)
        active, silent = [], 0
        start = None
        while not stop.is_set():
            frame = proc.stdout.read(640)
            if len(frame) != 640:
                err = proc.stderr.read(500).decode("utf-8", "replace") if proc.poll() is not None else "capture stream ended"
                raise RuntimeError("arecord capture failed: " + err)
            voiced = vad.is_speech(frame, 16000)
            if voiced:
                start = start or time.monotonic()
                active.append(frame)
                silent = 0
            elif active:
                active.append(frame)
                silent += 1
                if silent >= 15:
                    pcm = b"".join(active[:-15])
                    if pcm:
                        transcript = _whisper(pcm, model, binary)
                        events.put({"type": "transcript", "at": time.monotonic(), "vad_at": start,
                                    "transcript": transcript, "source": "microphone:" + device})
                    active, silent, start = [], 0, None
    except Exception as exc:
        events.put({"type": "error", "source": "audio", "error": str(exc), "trace": traceback.format_exc()})
    finally:
        if proc is not None:
            _stop_subprocess(proc)


def _stop_subprocess(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def _llm(prompt: str, url: str, n_predict: int, opener=request.urlopen, sleep=time.sleep):
    body = json.dumps({"prompt": prompt, "temperature": 0, "n_predict": n_predict, "stream": True}).encode()
    req = request.Request(url.rstrip("/") + "/completion", data=body, headers={"Content-Type": "application/json"})
    first = None
    text = []
    start = time.monotonic()
    for attempt in range(3):
        try:
            response = opener(req, timeout=120)
            break
        except urlerror.HTTPError as exc:
            if exc.code != 503 or attempt == 2:
                raise
            sleep(0.5 * (2 ** attempt))
    with response as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                item = json.loads(data)
            except json.JSONDecodeError:
                continue
            piece = item.get("content", "")
            if piece:
                first = first or time.monotonic()
                text.append(piece)
    return "".join(text), (first - start if first else None), time.monotonic() - start


def _ram_used_mb() -> float:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                values[k] = int(v.split()[0])
        return (values["MemTotal"] - values["MemAvailable"]) / 1024
    except (OSError, KeyError, ValueError):
        return 0


def _cleanup_children():
    if _ACTIVE_STOP is not None:
        _ACTIVE_STOP.set()
    for proc in _ACTIVE_CHILDREN:
        if proc.is_alive():
            proc.join(timeout=5)
        if proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                proc.terminate()
            proc.join(timeout=3)
        if proc.is_alive() and hasattr(proc, "kill"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            proc.join(timeout=3)
    _ACTIVE_CHILDREN.clear()


def _run_session(args):
    _require_safe_device()
    temps = _temps()
    while temps and max(temps) >= 78:
        print("Thermal safety pause: max %.1f C; waiting until below 70 C" % max(temps), flush=True)
        time.sleep(2)
        temps = _temps()
    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / ("turns-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")
    events = mp.Queue()
    stop = mp.Event()
    global _ACTIVE_STOP
    _ACTIVE_STOP = stop
    images = sorted(glob.glob(args.images))
    audio = [p for pat in args.audio_files for p in glob.glob(pat)]
    vision_args = (events, stop, args.yolo_engine, images, args.vision_fps, args.camera_index)
    if args.audio_device:
        audio_target = _live_audio_proc
        audio_args = (events, stop, args.audio_device, args.whisper_model, args.whisper_bin, args.vad_mode)
    else:
        audio_target = _audio_proc
        audio_args = (events, stop, audio, args.whisper_model, args.whisper_bin, args.vad_mode)
    vision = mp.Process(target=_process_entry, args=(_vision_proc, vision_args), name="vision")
    audio_proc = mp.Process(target=_process_entry, args=(audio_target, audio_args), name="audio")
    _ACTIVE_CHILDREN.clear()
    vision.start()
    _ACTIVE_CHILDREN.append(vision)
    audio_proc.start()
    _ACTIVE_CHILDREN.append(audio_proc)
    detections, last_transcript = [], ""
    max_temp, peak_ram, fps_values = 0.0, 0.0, []
    turns = errors = 0
    deadline = time.monotonic() + args.duration
    with log_path.open("a", buffering=1) as log:
        while time.monotonic() < deadline:
            temps = _temps()
            max_temp = max([max_temp] + temps)
            peak_ram = max(peak_ram, _ram_used_mb())
            if temps and max(temps) >= 78:
                while time.monotonic() < deadline and max(_temps() or [0]) >= 70:
                    time.sleep(2)
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event["type"] == "vision":
                detections = event["detections"]
            elif event["type"] == "fps":
                fps_values.append(event["fps"])
            elif event["type"] == "error":
                errors += 1
                log.write(json.dumps({"kind": "error", **event, "time_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            elif event["type"] == "transcript":
                last_transcript = event["transcript"]
                vad_to_transcript = event["at"] - event["vad_at"]
                prompt = build_prompt(_self_state(detections, last_transcript), detections, last_transcript)
                predict = 48 if temps and max(temps) >= 75 else args.n_predict
                t0 = time.monotonic()
                llm_started = None
                try:
                    llm_started = time.monotonic()
                    answer, ttft, llm_done = _llm(prompt, args.llm_url, predict)
                    turns += 1
                    rec = {"kind": "turn", "turn": turns, "time_utc": datetime.now(timezone.utc).isoformat(),
                           "audio_source": event.get("source"), "transcript": last_transcript, "response": answer,
                           "vad_to_transcript_s": round(vad_to_transcript, 3), "transcript_to_llm_s": round(time.monotonic() - t0, 3),
                           "transcript_to_llm_first_token_s": round(llm_started - t0 + ttft, 3) if ttft is not None else None,
                           "llm_first_token_s": round(ttft, 3) if ttft is not None else None, "llm_done_s": round(llm_done, 3),
                           "turn_latency_s": round(vad_to_transcript + time.monotonic() - t0, 3),
                           "n_predict": predict, "detections": detections, "max_temp_c_so_far": max_temp,
                           "peak_ram_mb_so_far": peak_ram}
                except Exception as exc:
                    errors += 1
                    rec = {"kind": "error", "source": "llm", "error": str(exc), "transcript": last_transcript}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = {"kind": "run_summary", "duration_s": args.duration, "turns_completed": turns,
                   "vision_fps_mean": round(sum(fps_values) / len(fps_values), 3) if fps_values else 0,
                   "peak_ram_mb": round(peak_ram, 1), "max_temp_c": max_temp, "errors": errors,
                   "log_path": str(log_path), "audio_inputs": ["microphone:" + args.audio_device] if args.audio_device else audio,
                   "image_inputs": len(images)}
        log.write(json.dumps(summary) + "\n")
    _cleanup_children()
    _ACTIVE_STOP = None
    print(json.dumps(summary, indent=2))
    return 0 if turns > 0 and errors == 0 else 1


def _run(args):
    lock_fd = _acquire_run_lock()
    try:
        _require_safe_device()
        _wait_for_health(args.llm_url)
        return _run_session(args)
    finally:
        _cleanup_children()
        os.close(lock_fd)


def _exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-engine", default="/models/vision/yolov8n-dla-fp16.engine")
    parser.add_argument("--images", default="/models/coco350/val2017/*")
    parser.add_argument("--whisper-model", default="/models/models/ggml-tiny.en.bin")
    parser.add_argument("--whisper-bin", default="whisper-cli")
    parser.add_argument("--audio-files", nargs="+", default=["/models/audio/jfk.wav", "/models/audio/question.wav"])
    parser.add_argument("--audio-device", default=None, help="live arecord device, for example hw:1,0")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--log-dir", default="/models/proto/logs")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--vision-fps", type=float, default=5)
    parser.add_argument("--camera-index", type=int, default=None, help="use a live OpenCV camera instead of COCO files")
    parser.add_argument("--n-predict", type=int, default=96)
    parser.add_argument("--vad-mode", type=int, choices=range(4), default=2)
    args = parser.parse_args()
    if args.duration <= 0 or args.vision_fps <= 0:
        parser.error("duration and vision FPS must be positive")
    if args.camera_index is None and glob.glob("/dev/video*"):
        args.camera_index = 0
    if args.audio_device is None:
        try:
            devices = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=3).stdout
            external = next((int(match.group(1)) for line in devices.splitlines()
                             if "card " in line and not re.search(r"\bAPE\b|NVIDIA Jetson NX", line, re.I)
                             for match in [re.search(r"card\s+(\d+):", line)] if match), None)
            if external is not None:
                args.audio_device = "hw:%d,0" % external
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not args.audio_device and "/models/audio/question.wav" in args.audio_files and not Path("/models/audio/question.wav").exists() and shutil.which("espeak-ng"):
        Path("/models/audio").mkdir(parents=True, exist_ok=True)
        subprocess.run(["espeak-ng", "-w", "/models/audio/question.wav", "What can you see and how are you doing?"], check=True)
    return _run(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())
