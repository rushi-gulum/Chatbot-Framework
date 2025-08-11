"""
Voice Channel Implementation for Universal Chatbot Framework

This module provides voice channel implementation for handling voice calls and
speech-to-text/text-to-speech interactions through various providers.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid
import tempfile
import os

import speech_recognition as sr
from gtts import gTTS
import io
import base64
from pydub import AudioSegment
from pydub.playback import play

from .base_channel import (
    BaseChannel, ChannelMessage, ChannelResponse, ChannelType, 
    MessageType, ChannelUser, MessageAttachment, ChannelError
)

# Configure logging
logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages a voice conversation session."""
    
    def __init__(self, session_id: str, user_id: str, call_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.call_id = call_id
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_audio_buffer = None
        
    def add_to_history(self, speaker: str, content: str, timestamp: Optional[datetime] = None):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "speaker": speaker,
            "content": content,
            "timestamp": (timestamp or datetime.utcnow()).isoformat()
        })
        self.last_activity = datetime.utcnow()
    
    def get_conversation_duration(self) -> float:
        """Get conversation duration in seconds."""
        return (datetime.utcnow() - self.started_at).total_seconds()


class VoiceChannel(BaseChannel):
    """
    Voice channel implementation for speech-based interactions.
    
    Supports speech-to-text, text-to-speech, and voice call management
    through various providers and audio processing libraries.
    """
    
    def __init__(self, channel_config: Dict[str, Any]):
        """
        Initialize the voice channel.
        
        Args:
            channel_config: Configuration including STT/TTS providers, audio settings, etc.
        """
        super().__init__(channel_config)
        
        # Speech Recognition Configuration
        self.stt_provider = channel_config.get("stt_provider", "google")  # google, azure, aws, whisper
        self.stt_language = channel_config.get("stt_language", "en-US")
        self.stt_config = channel_config.get("stt_config", {})
        
        # Text-to-Speech Configuration
        self.tts_provider = channel_config.get("tts_provider", "gtts")  # gtts, azure, aws, elevenlabs
        self.tts_language = channel_config.get("tts_language", "en")
        self.tts_voice = channel_config.get("tts_voice", "en-us-standard-A")
        self.tts_config = channel_config.get("tts_config", {})
        
        # Audio Configuration
        self.audio_format = channel_config.get("audio_format", "wav")
        self.sample_rate = channel_config.get("sample_rate", 16000)
        self.chunk_size = channel_config.get("chunk_size", 1024)
        self.max_recording_duration = channel_config.get("max_recording_duration", 30)  # seconds
        
        # Call Management
        self.telephony_provider = channel_config.get("telephony_provider")  # twilio, vonage, etc.
        self.phone_number = channel_config.get("phone_number")
        self.webhook_url = channel_config.get("webhook_url")
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Session management
        self.active_sessions: Dict[str, VoiceSession] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Audio processing
        self.temp_audio_dir = tempfile.mkdtemp(prefix="voice_channel_")
    
    def _get_channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.VOICE
    
    async def initialize(self) -> bool:
        """Initialize the voice channel."""
        try:
            # Initialize microphone for local testing
            try:
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize microphone: {e}")
            
            # Test TTS functionality
            try:
                test_audio = await self._text_to_speech("Voice channel initialized", test_mode=True)
                if test_audio:
                    logger.info("TTS functionality verified")
            except Exception as e:
                logger.warning(f"TTS test failed: {e}")
            
            self.is_active = True
            logger.info("Voice channel initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice channel: {e}")
            return False
    
    async def send_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """
        Send a voice response to the recipient.
        
        Args:
            response: The response to send
            recipient_id: Session ID or phone number
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            # Format response for voice channel
            formatted_response = self.format_response_for_channel(response)
            
            # Convert text to speech
            audio_data = await self._text_to_speech(formatted_response.content)
            if not audio_data:
                logger.error("Failed to convert text to speech")
                return False
            
            # Get or create session
            session = self.active_sessions.get(recipient_id)
            if not session:
                logger.warning(f"No active session found for recipient: {recipient_id}")
                return False
            
            # Add to conversation history
            session.add_to_history("bot", formatted_response.content)
            
            # Send audio based on session type
            if session.call_id:
                # Send via telephony provider
                return await self._send_call_audio(session.call_id, audio_data)
            else:
                # Send via WebRTC or other real-time method
                return await self._send_realtime_audio(session.session_id, audio_data)
                
        except Exception as e:
            logger.error(f"Error sending voice message: {e}")
            return False
    
    async def _text_to_speech(self, text: str, test_mode: bool = False) -> Optional[bytes]:
        """Convert text to speech audio."""
        try:
            if self.tts_provider == "gtts":
                return await self._gtts_synthesis(text, test_mode)
            else:
                logger.error(f"Unsupported TTS provider: {self.tts_provider}")
                return None
                
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return None
    
    async def _gtts_synthesis(self, text: str, test_mode: bool = False) -> Optional[bytes]:
        """Synthesize speech using Google Text-to-Speech."""
        try:
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=self.tts_language,
                slow=self.tts_config.get("slow", False)
            )
            
            # Save to temporary file
            temp_file = os.path.join(self.temp_audio_dir, f"tts_{uuid.uuid4()}.mp3")
            tts.save(temp_file)
            
            # Read audio data
            with open(temp_file, "rb") as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.remove(temp_file)
            
            if test_mode:
                logger.info("TTS synthesis test completed successfully")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in gTTS synthesis: {e}")
            return None
    
    async def _speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech audio to text."""
        try:
            # Save audio data to temporary file
            temp_file = os.path.join(self.temp_audio_dir, f"stt_{uuid.uuid4()}.wav")
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # Load audio with speech_recognition
            with sr.AudioFile(temp_file) as source:
                audio = self.recognizer.record(source)
            
            # Perform speech recognition
            try:
                if self.stt_provider == "google":
                    # Use getattr to safely access the method
                    recognize_method = getattr(self.recognizer, 'recognize_google', None)
                    if recognize_method:
                        text = recognize_method(
                            audio,
                            language=self.stt_language
                        )
                    else:
                        logger.error("Google speech recognition not available. Please install the required dependencies.")
                        return None
                else:
                    logger.error(f"Unsupported STT provider: {self.stt_provider}")
                    return None
            except AttributeError:
                logger.error("Speech recognition method not available. Please install the speech_recognition library.")
                return None
            except Exception as e:
                logger.error(f"Speech recognition failed: {e}")
                return None
            
            # Clean up temporary file
            os.remove(temp_file)
            
            logger.info(f"Speech recognition result: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return None
    
    async def receive_message(self) -> Optional[ChannelMessage]:
        """Receive a message from the message queue."""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving voice message: {e}")
            return None
    
    async def process_audio_input(self, audio_data: bytes, user_id: str, session_id: Optional[str] = None) -> Optional[ChannelMessage]:
        """
        Process incoming audio input and convert to text.
        
        Args:
            audio_data: Raw audio data
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Optional[ChannelMessage]: Parsed message or None
        """
        try:
            # Convert speech to text
            text_content = await self._speech_to_text(audio_data)
            if not text_content:
                return None
            
            # Get or create session
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session = self.active_sessions.get(session_id)
            if not session:
                session = VoiceSession(session_id, user_id)
                self.active_sessions[session_id] = session
            
            # Add to conversation history
            session.add_to_history("user", text_content)
            
            # Create audio attachment
            audio_attachment = MessageAttachment(
                attachment_type=MessageType.AUDIO,
                url="",  # Could store in cloud storage
                filename=f"voice_input_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav",
                data=audio_data,
                mime_type="audio/wav"
            )
            
            # Create channel user
            user = ChannelUser(
                user_id=user_id,
                channel_user_id=user_id
            )
            
            # Create channel message
            message = ChannelMessage(
                message_id=str(uuid.uuid4()),
                user=user,
                content=text_content,
                message_type=MessageType.AUDIO,
                channel_type=ChannelType.VOICE,
                session_id=session_id,
                attachments=[audio_attachment],
                metadata={
                    "original_audio_length": len(audio_data),
                    "speech_confidence": 0.95,  # Could get from STT service
                    "voice_session_id": session_id
                }
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            return None
    
    def _get_audio_duration(self, audio_data: bytes) -> float:
        """Get duration of audio in seconds."""
        try:
            # Create temporary file to analyze
            temp_file = os.path.join(self.temp_audio_dir, f"temp_{uuid.uuid4()}.wav")
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # Use pydub to get duration
            audio = AudioSegment.from_wav(temp_file)
            duration = len(audio) / 1000.0  # Convert to seconds
            
            # Clean up
            os.remove(temp_file)
            
            return duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0
    
    async def start_call_session(self, call_id: str, caller_id: str) -> str:
        """
        Start a new voice call session.
        
        Args:
            call_id: Telephony provider call ID
            caller_id: Phone number or user ID of caller
            
        Returns:
            str: Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            session = VoiceSession(session_id, caller_id, call_id)
            self.active_sessions[session_id] = session
            
            logger.info(f"Started voice call session {session_id} for call {call_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting call session: {e}")
            return ""
    
    async def end_call_session(self, session_id: str) -> bool:
        """
        End a voice call session.
        
        Args:
            session_id: Session to end
            
        Returns:
            bool: True if ended successfully
        """
        try:
            session = self.active_sessions.get(session_id)
            if session:
                session.is_active = False
                duration = session.get_conversation_duration()
                
                logger.info(f"Ended voice session {session_id}, duration: {duration:.2f}s")
                
                # Store conversation history if needed
                await self._store_conversation_history(session)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error ending call session: {e}")
            return False
    
    async def _store_conversation_history(self, session: VoiceSession):
        """Store conversation history for analytics/training."""
        try:
            # This could store to database, file, or analytics service
            history_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "call_id": session.call_id,
                "started_at": session.started_at.isoformat(),
                "duration": session.get_conversation_duration(),
                "conversation": session.conversation_history
            }
            
            # For now, just log the history
            logger.info(f"Storing conversation history for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error storing conversation history: {e}")
    
    async def _send_call_audio(self, call_id: str, audio_data: bytes) -> bool:
        """Send audio through telephony provider."""
        try:
            # This would integrate with telephony providers like Twilio
            # For now, simulate success
            logger.info(f"Sending audio to call {call_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending call audio: {e}")
            return False
    
    async def _send_realtime_audio(self, session_id: str, audio_data: bytes) -> bool:
        """Send audio through real-time connection (WebRTC, etc.)."""
        try:
            # This would integrate with WebRTC or similar real-time audio
            # For now, simulate success
            logger.info(f"Sending real-time audio to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending real-time audio: {e}")
            return False
    
    async def validate_message(self, message: ChannelMessage) -> bool:
        """Validate an incoming voice message."""
        try:
            # Check for audio content or text transcription
            if not message.content and not message.attachments:
                logger.warning("Voice message has no content or attachments")
                return False
            
            # Validate session
            if message.session_id and message.session_id not in self.active_sessions:
                logger.warning(f"Invalid session ID: {message.session_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating voice message: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get voice channel capabilities."""
        return [
            "speech_to_text",
            "text_to_speech",
            "audio_attachments",
            "voice_calls",
            "real_time_audio",
            "conversation_history",
            "audio_recording",
            "multiple_languages"
        ]
    
    def format_response_for_channel(self, response: ChannelResponse) -> ChannelResponse:
        """Format response for voice channel requirements."""
        # Remove markdown and HTML formatting for speech
        content = response.content
        content = content.replace("**", "").replace("*", "")
        content = content.replace("__", "").replace("_", "")
        content = content.replace("```", "").replace("`", "")
        content = content.replace("#", "")
        
        # Add pauses for better speech flow
        content = content.replace(".", ". ").replace("!", "! ").replace("?", "? ")
        content = content.replace("\n", " ... ")
        
        # Limit length for voice responses
        if len(content) > 500:
            content = content[:497] + "..."
        
        response.content = content
        response.channel_type = ChannelType.VOICE
        return response
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a voice session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "call_id": session.call_id,
                "started_at": session.started_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "duration": session.get_conversation_duration(),
                "is_active": session.is_active,
                "conversation_length": len(session.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return None
    
    async def close(self) -> bool:
        """Close the voice channel and cleanup resources."""
        try:
            # End all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.end_call_session(session_id)
            
            # Clean up temporary audio directory
            import shutil
            if os.path.exists(self.temp_audio_dir):
                shutil.rmtree(self.temp_audio_dir)
            
            self.is_active = False
            logger.info("Voice channel closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing voice channel: {e}")
            return False