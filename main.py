#!/usr/bin/env python3
"""
Minimal RPi2W Voice Assistant.
- Wake word: "mary" (case-insensitive, only at start of prompt)
- Records with arecord, no PyAudio, no Picovoice
- Supports Bangla and English prompts (Google STT: bn-BD + en-US)
- 45s session window after wake word, resets on each valid prompt
- AI: Groq, TTS: ElevenLabs (playback with aplay)
"""

import os, time, tempfile, subprocess, requests

# === CONFIG ===
WAKE_WORD = "mary"
STOP_WORDS = ["stop", "goodbye", "cancel"]
ACTIVE_WINDOW_SEC = 45
LLM_API_KEY = "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"
ELEVENLABS_API_KEY = "sk_62acd737ac494e5458ba36b979597c657f1ce30fd781664b"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
GOOGLE_CLOUD_CREDENTIALS_JSON = "./ancient-snow-462602-e7-e74438b65c52.json"

# === RECORDING ===
def record_audio(out_wav, max_seconds=8):
    print("🎤 Listening... (max {}s)".format(max_seconds))
    subprocess.run([
        "arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-q",
        "-d", str(max_seconds), out_wav
    ], check=True)

# === GOOGLE STT (BN + EN) ===
def stt_google_wav(wav_file_path):
    from google.cloud import speech_v1p1beta1 as speech
    import io
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_CREDENTIALS_JSON
    client = speech.SpeechClient()
    with io.open(wav_file_path, "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="bn-BD",  # Bangla
        alternative_language_codes=["en-US"],  # English
        enable_automatic_punctuation=True
    )
    resp = client.recognize(config=config, audio=audio)
    if not resp.results: return ""
    return resp.results[0].alternatives[0].transcript.strip()

# === AI & TTS ===
def ai_response(text):
    prompt = "You are a helpful, concise AI assistant for Bangla and English speakers. Reply in the user's language. Speak clearly for TTS."
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "system", "content": prompt},{"role": "user", "content": text}],
            "temperature": 0.2, "max_tokens": 120, "top_p": 0.8
        }, timeout=10)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def tts_elevenlabs(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    json_data = {
        "text": text[:1000],
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.7, "use_speaker_boost": True},
        "model_id": "eleven_turbo_v2"
    }
    r = requests.post(url, headers=headers, json=json_data, timeout=15)
    r.raise_for_status()
    return r.content

def play_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        tf.write(audio_bytes)
        path = tf.name
    subprocess.run(['aplay', '-q', path])
    os.unlink(path)

# === WAKE WORD DETECTION (NO PICOVOICE) ===
def heard_wake_word(text):
    return text.lower().startswith(WAKE_WORD)

def heard_stop_word(text):
    return any(word in text.lower() for word in STOP_WORDS)

# === MAIN LOGIC ===
def wait_for_wake_word():
    print("💤  Say 'mary' to wake me up...")
    while True:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            wav_path = tf.name
        record_audio(wav_path, max_seconds=3)
        text = stt_google_wav(wav_path)
        os.unlink(wav_path)
        if heard_wake_word(text):
            print(f"👂 Wake word detected in: {text}")
            return
        else:
            print(f"❌ Not a wake word: {text}")

def session_loop():
    print("🟢 Active for 45 seconds. Give me your prompt (Bangla or English).")
    session_end = time.time() + ACTIVE_WINDOW_SEC
    while time.time() < session_end:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            wav_path = tf.name
        record_audio(wav_path, max_seconds=8)
        text = stt_google_wav(wav_path)
        os.unlink(wav_path)
        if not text.strip():
            print("🤫 No speech detected.")
            continue
        print(f"📝 Heard: {text}")
        if heard_stop_word(text):
            print("🛑 Stop word detected. Going to sleep.")
            return
        ai_text = ai_response(text)
        print(f"🤖 AI: {ai_text}")
        audio = tts_elevenlabs(ai_text)
        play_audio(audio)
        session_end = time.time() + ACTIVE_WINDOW_SEC  # Reset timer after valid prompt

    print("⏰ Session ended. Say 'mary' to wake me again.")

def main_loop():
    while True:
        wait_for_wake_word()
        session_loop()

if __name__ == "__main__":
    main_loop()