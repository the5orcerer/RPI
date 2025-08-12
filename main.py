#!/usr/bin/env python3
"""
Combined Voice Assistant API with Integrated Google STT
Runs everything on RPi - STT, AI Processing, TTS, Wake Word Detection
Blazing fast with minimal resource usage
"""

from fastapi import FastAPI, HTTPException, Request, Form, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import requests
import tempfile
import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import aiofiles
from pydantic import BaseModel
import uvicorn
import threading
import sys
import pyaudio
from collections import deque
from six.moves import queue
from google.cloud import speech
from concurrent.futures import ThreadPoolExecutor
import subprocess
import signal
import wave

# ================ CONFIGURATION ================
# Audio Configuration for STT
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Wake words and stop words
WAKE_WORDS = ["হ্যালো ম্যরি", "hey mary", "hello mary", "hi mary"]
STOP_WORDS = ["stop listening", "থামাও", "stop", "enough", "goodbye"]

# Audio settings
BUFFER_SECONDS = 3
MAX_RECORDING_SECONDS = 30
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2

# API Keys - Set your actual keys
LLM_API_KEY = "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"
ELEVENLABS_API_KEY = "sk_73a636adaa3b460751d57452edfa28ab27e62f9213f529ea"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Performance settings
API_TIMEOUT = 15

# ================ LOGGING ================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================ FASTAPI APP ================
app = FastAPI(
    title="Combined Voice Assistant API", 
    description="All-in-One Voice Assistant with STT + AI + TTS"
)

# Global storage
conversation_history: List[Dict[str, Any]] = []
system_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_processing_time": 0.0,
    "uptime_start": datetime.now(),
    "wake_word_detections": 0,
    "voice_mode_active": False
}

# Global voice assistant instance
voice_assistant = None

class TextRequest(BaseModel):
    text: str
    language: str = "en"
    return_audio: bool = True

class VoiceControlRequest(BaseModel):
    action: str  # "start", "stop", "status"

# ================ TEMPLATES ================
templates = Jinja2Templates(directory="templates")

def create_templates_dir():
    """Create templates directory and HTML files"""
    if not os.path.exists("templates"):
        os.makedirs("templates")
    
    admin_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Voice Assistant Control Panel</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); color: white; padding: 30px; border-radius: 15px; margin-bottom: 25px; text-align: center; }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }
        .voice-control { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; margin-bottom: 25px; text-align: center; }
        .voice-status { font-size: 1.2em; margin-bottom: 15px; }
        .status-active { color: #27ae60; font-weight: bold; }
        .status-inactive { color: #e74c3c; font-weight: bold; }
        .control-btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 12px 30px; border: none; border-radius: 25px; cursor: pointer; font-size: 16px; margin: 0 10px; transition: transform 0.2s; }
        .control-btn:hover { transform: translateY(-2px); }
        .control-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #667eea; }
        .conversations { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; }
        .conversation { border-left: 4px solid #667eea; padding: 15px; margin-bottom: 15px; background: rgba(102, 126, 234, 0.05); border-radius: 0 10px 10px 0; }
        .timestamp { color: #666; font-size: 0.9em; }
        .input-text { background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 5px 0; }
        .output-text { background: #e8f5e8; padding: 10px; border-radius: 8px; margin: 5px 0; }
        .test-section { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; margin-bottom: 25px; }
        .test-input { width: 70%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-right: 10px; }
        .api-info { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; margin-bottom: 25px; }
        .endpoint { background: #f8f9fa; padding: 8px 12px; border-radius: 5px; font-family: monospace; margin: 3px 0; }
    </style>
    <script>
        async function toggleVoiceMode() {
            const status = document.getElementById('voice-status');
            const btn = document.getElementById('voice-toggle-btn');
            
            btn.disabled = true;
            
            try {
                const currentActive = status.textContent.includes('ACTIVE');
                const action = currentActive ? 'stop' : 'start';
                
                const response = await fetch('/voice_control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: action})
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    location.reload();
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
            
            btn.disabled = false;
        }
        
        async function testTextProcessing() {
            const input = document.getElementById('test-text').value;
            if (!input.trim()) return;
            
            const resultDiv = document.getElementById('test-result');
            resultDiv.innerHTML = 'Processing...';
            
            try {
                const response = await fetch('/process_text', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: input, return_audio: false})
                });
                
                const result = await response.json();
                resultDiv.innerHTML = `<strong>AI Response:</strong> ${result.response_text}<br><strong>Time:</strong> ${result.processing_time.toFixed(2)}s`;
            } catch (error) {
                resultDiv.innerHTML = 'Error: ' + error.message;
            }
        }
        
        setInterval(() => location.reload(), 30000); // Auto refresh
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Combined Voice Assistant</h1>
            <p>STT + AI + TTS | Wake Word: "{{ wake_words }}" | Stop: "{{ stop_words }}"</p>
        </div>
        
        <div class="voice-control">
            <div class="voice-status" id="voice-status">
                Voice Mode: <span class="status-{{ 'active' if stats.voice_mode_active else 'inactive' }}">
                    {{ 'ACTIVE' if stats.voice_mode_active else 'INACTIVE' }}
                </span>
            </div>
            <button class="control-btn" id="voice-toggle-btn" onclick="toggleVoiceMode()">
                {{ 'Stop Voice Mode' if stats.voice_mode_active else 'Start Voice Mode' }}
            </button>
        </div>
        
        <div class="test-section">
            <h3>🧪 Quick Text Test</h3>
            <input type="text" id="test-text" class="test-input" placeholder="Type a message to test AI processing..." />
            <button class="control-btn" onclick="testTextProcessing()">Test</button>
            <div id="test-result" style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;"></div>
        </div>
        
        <div class="api-info">
            <h3>🔌 API Endpoints</h3>
            <div class="endpoint">POST /process_text - Process text input</div>
            <div class="endpoint">POST /voice_control - Control voice mode</div>
            <div class="endpoint">GET /health - System health</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.total_requests }}</div>
                <div>Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.successful_requests }}</div>
                <div>Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.wake_word_detections }}</div>
                <div>Wake Words</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ "%.2f"|format(stats.avg_processing_time) }}s</div>
                <div>Avg Time</div>
            </div>
        </div>
        
        <div class="conversations">
            <h2>💬 Recent Conversations</h2>
            {% if conversations %}
                {% for conv in conversations %}
                <div class="conversation">
                    <div class="timestamp">{{ conv.timestamp.strftime('%H:%M:%S') }} | {{ "%.3f"|format(conv.processing_time) }}s | {{ conv.language.upper() }}</div>
                    <div><strong>Input:</strong></div>
                    <div class="input-text">{{ conv.input_text }}</div>
                    <div><strong>Response:</strong></div>
                    <div class="output-text">{{ conv.output_text }}</div>
                </div>
                {% endfor %}
            {% else %}
                <p>No conversations yet. Try the text test above or activate voice mode!</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
    '''
    
    with open("templates/admin.html", "w") as f:
        f.write(admin_html)

create_templates_dir()

# ================ STT INTEGRATION ================

class OptimizedMicrophoneStream:
    """Optimized microphone stream for STT"""
    
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue(maxsize=50)
        self.closed = True
        self._audio_interface = None
        self._audio_stream = None

    def __enter__(self):
        try:
            self._audio_interface = pyaudio.PyAudio()
            device_index = self._get_default_input_device()
            
            self._audio_stream = self._audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self._rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self._chunk,
                stream_callback=self._fill_buffer,
                start=False
            )
            
            self._audio_stream.start_stream()
            self.closed = False
            return self
            
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            raise

    def __exit__(self, type, value, traceback):
        self._cleanup()

    def _get_default_input_device(self):
        try:
            return self._audio_interface.get_default_input_device_info()['index']
        except:
            return None

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        if not self._buff.full():
            self._buff.put(in_data)
        return None, pyaudio.paContinue

    def _cleanup(self):
        try:
            if self._audio_stream:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            if self._audio_interface:
                self._audio_interface.terminate()
            self.closed = True
        except Exception as e:
            logger.error(f"Audio cleanup error: {e}")

    def generator(self):
        while not self.closed:
            try:
                chunk = self._buff.get(timeout=1)
                if chunk is None:
                    return
                yield chunk
            except queue.Empty:
                continue

class IntegratedVoiceAssistant:
    """Voice assistant integrated with FastAPI"""
    
    def __init__(self):
        self.running = False
        self.wake_word_detected = False
        self.speech_client = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Google Speech
        try:
            self.speech_client = speech.SpeechClient()
            logger.info("✅ Google Speech client ready")
        except Exception as e:
            logger.error(f"❌ Google Speech init failed: {e}")

    def play_audio_system(self, audio_data):
        """Play audio using system commands"""
        def _play():
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_path = tmp_file.name
                
                if os.system("which aplay > /dev/null 2>&1") == 0:
                    subprocess.run(['aplay', '-q', tmp_path], capture_output=True)
                elif os.system("which paplay > /dev/null 2>&1") == 0:
                    subprocess.run(['paplay', tmp_path], capture_output=True)
                
                os.unlink(tmp_path)
                
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
        
        threading.Thread(target=_play, daemon=True).start()

    async def process_with_api(self, text):
        """Process text using internal API functions"""
        try:
            # Get AI response
            ai_response = await get_ai_response(text)
            
            # Generate speech
            audio_data = await generate_speech(ai_response)
            
            # Play audio
            self.play_audio_system(audio_data)
            
            # Log conversation
            log_conversation(text, ai_response, "voice", "success", 1.0)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return None

    def listen_and_respond(self, responses):
        """Main listening loop"""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            transcript_lower = transcript.lower()

            if not self.wake_word_detected:
                # Wake word detection
                if any(wake_word.lower() in transcript_lower for wake_word in WAKE_WORDS):
                    self.wake_word_detected = True
                    system_stats["wake_word_detections"] += 1
                    logger.info(f"🎯 Wake word: {transcript}")
                    print("👂 Listening...")
                    
            else:
                # Active listening
                if result.is_final:
                    print(f"🎤 {transcript}")
                    
                    # Check stop words
                    if any(stop_word.lower() in transcript_lower for stop_word in STOP_WORDS):
                        self.wake_word_detected = False
                        logger.info("🛑 Stop detected")
                        print("😴 Sleeping...")
                    else:
                        # Process with API
                        asyncio.create_task(self.process_with_api(transcript))

    async def start_voice_mode(self):
        """Start voice recognition mode"""
        if self.running:
            return False
            
        self.running = True
        system_stats["voice_mode_active"] = True
        
        def voice_loop():
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="bn-BD",
                alternative_language_codes=["en-US"],
                enable_automatic_punctuation=True,
                model="latest_short"
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            logger.info("🎤 Voice mode started")
            print("😴 Say wake word to activate...")
            
            while self.running:
                try:
                    with OptimizedMicrophoneStream(RATE, CHUNK) as stream:
                        audio_generator = stream.generator()
                        requests = (
                            speech.StreamingRecognizeRequest(audio_content=content)
                            for content in audio_generator
                        )
                        
                        responses = self.speech_client.streaming_recognize(
                            streaming_config, requests
                        )
                        
                        self.listen_and_respond(responses)
                        
                except Exception as e:
                    if self.running:
                        logger.error(f"Voice loop error: {e}")
                        time.sleep(2)
        
        # Start voice loop in background
        self.executor.submit(voice_loop)
        return True

    def stop_voice_mode(self):
        """Stop voice recognition mode"""
        self.running = False
        system_stats["voice_mode_active"] = False
        logger.info("🛑 Voice mode stopped")
        return True

# ================ AI + TTS FUNCTIONS ================

async def get_ai_response(text: str, language: str = "en") -> str:
    """Get AI response from Groq"""
    try:
        system_prompt = (
            "You are Zoya, a helpful AI assistant. Give concise, friendly responses "
            "optimized for voice interaction. Keep responses under 2 sentences when possible."
        )
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "max_tokens": 150,  # Short for voice
                "top_p": 0.8
            },
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code}")
        
        return response.json()["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        logger.error(f"AI response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_speech(text: str) -> bytes:
    """Generate speech using ElevenLabs"""
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        
        json_data = {
            "text": text[:1000],  # Limit for speed
            "voice_settings": {
                "stability": 0.8,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "model_id": "eleven_turbo_v2"
        }
        
        response = requests.post(url, headers=headers, json=json_data, timeout=15)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"ElevenLabs error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def log_conversation(input_text: str, output_text: str, language: str, status: str, processing_time: float):
    """Log conversation"""
    conversation_history.append({
        "timestamp": datetime.now(),
        "input_text": input_text,
        "output_text": output_text,
        "language": language,
        "status": status,
        "processing_time": processing_time
    })
    
    # Keep last 20
    if len(conversation_history) > 20:
        conversation_history.pop(0)
    
    # Update stats
    if status == "success":
        system_stats["successful_requests"] += 1
        # Update average
        times = [c["processing_time"] for c in conversation_history if c["status"] == "success"]
        if times:
            system_stats["avg_processing_time"] = sum(times) / len(times)

# ================ API ENDPOINTS ================

@app.get("/", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin control panel"""
    uptime_hours = (datetime.now() - system_stats["uptime_start"]).total_seconds() / 3600
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "stats": system_stats,
        "uptime": f"{uptime_hours:.1f}",
        "conversations": list(reversed(conversation_history)),
        "wake_words": ", ".join(WAKE_WORDS[:2]),
        "stop_words": ", ".join(STOP_WORDS[:2])
    })

@app.post("/process_text")
async def process_text_endpoint(request: TextRequest):
    """Process text input"""
    start_time = time.time()
    
    try:
        system_stats["total_requests"] += 1
        
        # Detect language
        language = "bn" if any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in request.text) else "en"
        
        # Get AI response
        ai_response = await get_ai_response(request.text, language)
        
        processing_time = time.time() - start_time
        
        # Log conversation
        log_conversation(request.text, ai_response, language, "success", processing_time)
        
        if request.return_audio:
            # Generate speech
            audio_data = await generate_speech(ai_response)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(temp_file.name, "wb") as f:
                f.write(audio_data)
            
            return FileResponse(
                temp_file.name,
                media_type="audio/wav",
                filename="response.wav",
                headers={
                    "X-Response-Text": ai_response,
                    "X-Processing-Time": str(processing_time)
                },
                background=BackgroundTasks().add_task(lambda: os.unlink(temp_file.name))
            )
        else:
            return {
                "response_text": ai_response,
                "language": language,
                "processing_time": processing_time,
                "status": "success"
            }
            
    except Exception as e:
        system_stats["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_control")
async def voice_control_endpoint(request: VoiceControlRequest):
    """Control voice mode"""
    global voice_assistant
    
    try:
        if request.action == "start":
            if not voice_assistant:
                voice_assistant = IntegratedVoiceAssistant()
            
            success = await voice_assistant.start_voice_mode()
            return {"status": "success" if success else "error", "message": "Voice mode started" if success else "Already running"}
            
        elif request.action == "stop":
            if voice_assistant:
                voice_assistant.stop_voice_mode()
            return {"status": "success", "message": "Voice mode stopped"}
            
        elif request.action == "status":
            return {
                "status": "success", 
                "voice_active": system_stats["voice_mode_active"],
                "wake_words": WAKE_WORDS,
                "stop_words": STOP_WORDS
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Combined Voice Assistant API",
        "voice_mode_active": system_stats["voice_mode_active"],
        "uptime_hours": (datetime.now() - system_stats["uptime_start"]).total_seconds() / 3600,
        "stats": system_stats
    }

# ================ STARTUP ================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global voice_assistant
    logger.info("🚀 Combined Voice Assistant API starting...")
    logger.info(f"🎯 Wake words: {WAKE_WORDS}")
    logger.info(f"🛑 Stop words: {STOP_WORDS}")
    
    # Initialize voice assistant
    voice_assistant = IntegratedVoiceAssistant()

def signal_handler(signum, frame):
    """Graceful shutdown"""
    global voice_assistant
    logger.info("🛑 Shutting down...")
    if voice_assistant:
        voice_assistant.stop_voice_mode()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Combined Voice Assistant...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
