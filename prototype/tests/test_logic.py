from logic import build_prompt, dedupe_detections, pcm_frame_bytes, vad_chunks


def test_prompt_includes_live_self_state_and_audio_vision():
    prompt = build_prompt('{"thermal_zone_c":[61]}', ["[V:person:0.92]"], "hello")
    assert "<self-state>" in prompt
    assert '"thermal_zone_c":[61]' in prompt
    assert "[V:person:0.92]" in prompt
    assert "hello" in prompt


def test_detection_dedupe_reports_only_scene_changes():
    current, changed = dedupe_detections(None, ["[V:person:0.92]", "[V:person:0.92]"])
    assert current == ["[V:person:0.92]"]
    assert changed
    assert dedupe_detections(current, current) == (current, False)
    assert dedupe_detections(current, ["[V:car:0.84]"])[1]


def test_vad_chunker_produces_complete_20ms_mono_frames():
    assert pcm_frame_bytes(16000, 20) == 640
    pcm = bytes(range(256)) * 5
    chunks = vad_chunks(pcm, 16000, 20)
    assert len(chunks) == 2
    assert all(len(frame) == 640 for frame in chunks)
    assert vad_chunks(b"short", 16000) == []
