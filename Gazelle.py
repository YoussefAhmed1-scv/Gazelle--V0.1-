# Gazelle Brain - AI Voice Assistant
# Press Enter to activate Voice Mode

import os
import sys
import json
import queue
import msvcrt
import pyttsx3
import sounddevice as sd
from llama_cpp import Llama
from vosk import Model, KaldiRecognizer
from colorama import Fore, Style, init

# ---------------------- Init ----------------------
os.environ["VOSK_LOG_LEVEL"] = "-1"
init(autoreset=True)

# ---------------------- Load Config ----------------------
with open("config.json", "r") as f:
    config = json.load(f)

llama_model_path = config["llama_model_path"]
vosk_model_path = config["vosk_model_path"]
tts_rate = config["tts_rate"]
tts_volume = config["tts_volume"]
samplerate = config["samplerate"]
blocksize = config["blocksize"]
voice_mode_toggle_key = config["voice_mode_toggle_key"]

# ---------------------- Load Models ----------------------
llm = Llama(model_path=llama_model_path, n_ctx=2028, verbose=False)
vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, samplerate)
q = queue.Queue()

# ---------------------- TTS ----------------------
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", tts_rate)
    engine.setProperty("volume", tts_volume)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ---------------------- STT ----------------------
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def listen():
    """Listen to user and return text. Press Enter to toggle back to Text Mode."""
    with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize,
                           dtype='int16', channels=1, callback=callback):
        print(Fore.CYAN + f"🎤 Speak now... (Press {voice_mode_toggle_key} to switch to Text Mode)" + Style.RESET_ALL)
        while True:
            # Check for Enter key press
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'\r', b'\n'):
                    return "_TOGGLE_"
                if ch == b'\xe0' and msvcrt.kbhit():
                    ch2 = msvcrt.getch()
                    if ch2 == b'\x1c':
                        return "_TOGGLE_"

            # Get audio chunk
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    return text

# ---------------------- Main Loop ----------------------
voice_mode = False

while True:
    if voice_mode:
        user_input = listen()
        if user_input == "_TOGGLE_":
            voice_mode = False
            print(Fore.MAGENTA + "💬 Text Mode Activated" + Style.RESET_ALL)
            continue
        print(Fore.CYAN + f"You: {user_input}" + Style.RESET_ALL)
    else:
        user_input = input(Fore.CYAN + "You: " + Style.RESET_ALL)
        if user_input.strip() == "":
            voice_mode = True
            print(Fore.MAGENTA + "🔊 Voice Mode Activated" + Style.RESET_ALL)
            continue

    # Exit commands
    if user_input.lower() in ["bye", "exit", "سلام", "باي"]:
        break

    # Prepare prompt
    prompt = f"""You are Gazelle, a friendly and smart AI assistant.
Always reply in one single answer without repeating yourself,
and never say 'Assistant'.
Keep responses concise and helpful.

User: {user_input}
Gazelle:"""

    # Generate streaming response
    print(Fore.YELLOW + "Gazelle: " + Style.RESET_ALL, end="", flush=True)
    response_text = ""
    for token in llm(
        prompt=prompt,
        max_tokens=200,
        stop=["User:", "You:", "Gazelle:", "Assistant:"],
        stream=True
    ):
        chunk = token["choices"][0]["text"].replace("Assistant:", "")
        response_text += chunk
        print(Fore.YELLOW + chunk + Style.RESET_ALL, end="", flush=True)

    print()

    if voice_mode:
        speak(response_text)
