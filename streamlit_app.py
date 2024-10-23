import streamlit as st
import torch
from transformers import pipeline
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup page config
st.set_page_config(page_title="Real-time Transcription", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .transcription-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        min-height: 200px;
        margin-top: 20px;
    }
    .stStatus {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

# Create a placeholder for real-time updates
transcript_placeholder = st.empty()

# Load Whisper model
@st.cache_resource
def load_model():
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large",
        torch_dtype=torch.float16,
        device="cuda:0"
    )

pipe = load_model()

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"]}]}
)


class AudioProcessor:
    def __init__(self):
        self.audio_chunks = []
        self.transcript_buffer = ""
        self.sample_rate = 16000
        self.chunk_duration = 3  # Process every 3 seconds
        self.min_samples = self.sample_rate * self.chunk_duration
        logger.info("AudioProcessor initialized")

    def process_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # Convert audio frame to numpy array
            audio_array = frame.to_ndarray()
            logger.info(f"Received audio frame shape: {audio_array.shape}")

            # Append to chunks
            self.audio_chunks.extend(audio_array.flatten())
            
            # Process when we have enough samples
            if len(self.audio_chunks) >= self.min_samples:
                # Prepare audio chunk
                audio_data = np.array(self.audio_chunks[:self.min_samples])
                audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
                
                logger.info(f"Processing audio chunk of length: {len(audio_data)}")
                
                # Transcribe
                result = pipe(audio_data)
                text = result["text"].strip()
                
                if text:
                    logger.info(f"Transcribed text: {text}")
                    # Update transcript in session state
                    st.session_state.transcript += " " + text
                    # Update the placeholder with new text
                    transcript_placeholder.markdown(
                        f'<div class="transcription-box">{st.session_state.transcript}</div>',
                        unsafe_allow_html=True
                    )
                
                # Keep remaining samples
                self.audio_chunks = self.audio_chunks[self.min_samples:]
                
        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
        
        return frame

def create_audio_processor():
    return AudioProcessor()

# Main UI
st.title("Real-time Whisper Transcription")

# WebRTC audio streamer
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": False,
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
    },
    audio_processor_factory=create_audio_processor,
)

# Transcription display
st.markdown("### Live Transcription")
if not st.session_state.transcript:
    transcript_placeholder.markdown(
        '<div class="transcription-box">Transcription will appear here...</div>',
        unsafe_allow_html=True
    )

# Clear transcript button
col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    if st.button("Clear Transcript"):
        st.session_state.transcript = ""
        transcript_placeholder.markdown(
            '<div class="transcription-box">Transcription will appear here...</div>',
            unsafe_allow_html=True
        )

# Debugging audio input and streaming
if webrtc_ctx.state.playing:
    st.success("Audio streaming is active. Microphone input is being processed.")
    st.markdown("### Debug Information")
    st.write({
        "WebRTC State": webrtc_ctx.state.playing,
        "Audio Input": "Receiving",
        "Audio Processor Status": "Active",
        "Transcription Model": "Whisper Large",
        "Device": "CUDA" if torch.cuda.is_available() else "CPU"
    })
else:
    st.warning("Audio streaming is inactive. Ensure microphone permissions are enabled and click 'START'.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Powered by OpenAI Whisper</p>",
    unsafe_allow_html=True
)

