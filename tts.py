import os
import time
import datetime
import contextlib
import wave
import streamlit as st

# Data directory for TTS cache (kept local to this module)
DATA_DIR = "app_data"
TTS_CACHE_DIR = os.path.join(DATA_DIR, "tts_cache")

@st.cache_data
def get_available_voices():
    """Return list of available local TTS voice names (pyttsx3).
    If pyttsx3 isn't available, return an empty list."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        result = []
        for v in voices:
            name = getattr(v, 'name', None) or getattr(v, 'id', None)
            if name:
                result.append(name)
        try:
            engine.stop()
        except Exception:
            pass
        return result
    except Exception:
        return []

def get_audio_duration(path):
    """Return duration in seconds for a WAV file, or None on error."""
    try:
        with contextlib.closing(wave.open(path, 'r')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return None

def synthesize_to_file(text, filename=None, voice_name=None, rate=None):
    """Synthesize `text` to a WAV file using pyttsx3. Returns path or None on error.
    This function attempts to import pyttsx3; any errors are reported via Streamlit messages.
    """
    try:
        import pyttsx3
    except Exception:
        try:
            print("Local TTS engine (pyttsx3) is not installed or failed to import.")
            st.error("Local TTS engine (pyttsx3) is not installed or failed to import.")
        except Exception:
            print("ERROR")
            pass
        return None
    try:
        os.makedirs(TTS_CACHE_DIR, exist_ok=True)
        if not filename:
            filename = os.path.join(TTS_CACHE_DIR, f"tts_{int(time.time()*1000)}.wav")
        engine = pyttsx3.init()
        if rate:
            try:
                engine.setProperty('rate', int(rate))
            except Exception:
                pass
        if voice_name:
            try:
                voices = engine.getProperty('voices')
                for v in voices:
                    vname = getattr(v, 'name', None) or getattr(v, 'id', None)
                    if vname == voice_name or (voice_name in vname):
                        engine.setProperty('voice', v.id)
                        break
            except Exception:
                pass
        engine.save_to_file(text, filename)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        return filename
    except Exception as e:
        # Log traceback to a debug file for diagnosis
        try:
            os.makedirs(TTS_CACHE_DIR, exist_ok=True)
            import traceback
            with open(os.path.join(TTS_CACHE_DIR, 'tts_debug.log'), 'a', encoding='utf-8') as lf:
                lf.write(f"[{datetime.datetime.now().isoformat()}] TTS synthesis error: {e}\n")
                lf.write(traceback.format_exc())
        except Exception:
            pass
        try:
            st.error(f"TTS synthesis failed: {e}")
        except Exception:
            pass
        return None