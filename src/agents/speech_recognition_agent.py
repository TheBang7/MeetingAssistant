import os
from typing import Optional, Callable

import dashscope
import pyaudio
from dashscope.audio.asr import *
from dashscope.audio.asr import VocabularyService
from dotenv import load_dotenv

load_dotenv()


class SpeechRecognitionAgent:
    """
    Speech Recognition Agent, responsible for real-time transcription of microphone input speech to text
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 block_size: int = 3200,
                 format_pcm: str = 'pcm',
                 semantic_punctuation_enabled: bool = False,
                 on_text_callback: Optional[Callable[[str], None]] = None,
                 on_speech_complete_callback: Optional[Callable[[str], None]] = None,
                 ):
        """
        Initialize Speech Recognition Agent
        
        Args:
            sample_rate: Sample rate
            channels: Number of channels
            block_size: Buffer size
            format_pcm: Audio format
            semantic_punctuation_enabled: Whether to enable semantic punctuation
            on_text_callback: Text recognition callback function
            on_speech_complete_callback: Speech completion callback function
            keywords: List of keywords that need enhanced recognition
        """
        self.model = os.environ['DASHSCOPE_ASR_MODEL_NAME']
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.format_pcm = format_pcm
        self.semantic_punctuation_enabled = semantic_punctuation_enabled
        self.on_text_callback = on_text_callback
        self.on_speech_complete_callback = on_speech_complete_callback
        self.keywords = []
        # Get wake-up keyword from environment variable
        if 'WAKE_UP_KEYWORD' in os.environ:
            self.keywords.append(os.environ['WAKE_UP_KEYWORD'])

        # Vocabulary related properties
        self.vocabulary_service = None
        self.vocabulary_id = None

        self.mic = None
        self.stream = None
        self.recognition = None
        self.is_recording = False

        # Initialize API key
        self._init_api_key()

        self.mic = None
        self.stream = None
        self.recognition = None
        self.is_recording = False

        # Initialize API key
        self._init_api_key()

    def _init_api_key(self):
        """
        Initialize DashScope API key
        """
        if 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            print("Warning: DASHSCOPE_API_KEY environment variable not found, please ensure it is set correctly")

    def _create_vocabulary(self):
        """
        Create vocabulary
        """
        if not self.keywords:
            return None

        try:
            self.vocabulary_service = VocabularyService()
            # Prepare vocabulary data
            vocabulary_data = [
                {"text": keyword, "weight": 4, "lang": "zh"}
                for keyword in self.keywords
            ]

            # Create vocabulary
            print(f"Creating vocabulary, keywords: {self.keywords}")
            self.vocabulary_id = self.vocabulary_service.create_vocabulary(
                prefix="meeting",
                target_model=self.model,
                vocabulary=vocabulary_data
            )
            print(f"Vocabulary created successfully, ID: {self.vocabulary_id}")
            return self.vocabulary_id
        except Exception as e:
            print(f"Failed to create vocabulary: {e}")
            return None

    def _delete_vocabulary(self):
        """
        Delete vocabulary
        """
        if self.vocabulary_service and self.vocabulary_id:
            try:
                print(f"Deleting vocabulary: {self.vocabulary_id}")
                self.vocabulary_service.delete_vocabulary(self.vocabulary_id)
                print("Vocabulary deleted")
            except Exception as e:
                print(f"Failed to delete vocabulary: {e}")
            finally:
                self.vocabulary_id = None
                self.vocabulary_service = None

    def start_recognition(self):
        """
        Start speech recognition
        """
        if self.is_recording:
            print("Speech recognition is already running")
            return

        # Create callback handler
        callback = CustomRecognitionCallback(self)

        # Create vocabulary
        vocabulary_id = self._create_vocabulary()

        # Initialize recognition service
        self.recognition = Recognition(
            model=self.model,
            format=self.format_pcm,
            sample_rate=self.sample_rate,
            semantic_punctuation_enabled=self.semantic_punctuation_enabled,
            callback=callback,
            vocabulary_id=vocabulary_id,
            language_hints=['zh', 'en']
        )

        # Start recognition
        self.recognition.start()
        self.is_recording = True

        # Audio stream transmission is no longer set with signal processor here, it is the responsibility of the caller to stop
        print("Speech recognition started")

        # Start audio stream transmission
        self._start_audio_stream()

    def _start_audio_stream(self):
        """
        Start audio stream transmission
        """
        try:
            while self.is_recording:
                if self.stream:
                    data = self.stream.read(self.block_size, exception_on_overflow=False)
                    self.recognition.send_audio_frame(data)
                else:
                    break
        except Exception as e:
            print(f"Audio stream transmission error: {e}")
            self.stop_recognition()

    def stop_recognition(self):
        """
        Stop speech recognition
        """
        if not self.is_recording:
            return

        self.is_recording = False

        # Stop recognition service
        if self.recognition:
            self.recognition.stop()
            print('Speech recognition stopped')
            print(
                '[Metrics] requestId: {}, first package delay ms: {}, last package delay ms: {}'
                .format(
                    self.recognition.get_last_request_id(),
                    self.recognition.get_first_package_delay(),
                    self.recognition.get_last_package_delay(),
                )
            )

        # Clean up resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.mic:
            self.mic.terminate()
            self.mic = None

        # Delete vocabulary
        self._delete_vocabulary()

    def _signal_handler(self, sig, frame):
        """
        Signal handler to handle Ctrl+C interrupt
        """
        print('\nInterrupt signal detected, stopping speech recognition...')
        self.stop_recognition()

    def on_text_recognized(self, text: str):
        """
        Processing method when text is recognized
        
        Args:
            text: Recognized text
        """
        if self.on_text_callback:
            self.on_text_callback(text)
        else:
            print(f'Recognized text: {text}')

    def on_speech_complete(self, text: str):
        """
        Processing method when speech is complete
        
        Args:
            text: Complete speech text
        """
        if self.on_speech_complete_callback:
            self.on_speech_complete_callback(text)
        else:
            print(f'Speech complete: {text}')


class CustomRecognitionCallback(RecognitionCallback):
    """
    Custom recognition callback class for handling recognition events
    """

    def __init__(self, agent: SpeechRecognitionAgent):
        self.agent = agent

    def on_open(self) -> None:
        print('Recognition connection established')
        # Initialize microphone
        self.agent.mic = pyaudio.PyAudio()
        self.agent.stream = self.agent.mic.open(
            format=pyaudio.paInt16,
            channels=self.agent.channels,
            rate=self.agent.sample_rate,
            input=True
        )

    def on_close(self) -> None:
        print('Recognition connection closed')
        # Clean up microphone resources
        if self.agent.stream:
            self.agent.stream.stop_stream()
            self.agent.stream.close()
            self.agent.stream = None

        if self.agent.mic:
            self.agent.mic.terminate()
            self.agent.mic = None

    def on_complete(self) -> None:
        print('Recognition task completed')

    def on_error(self, message) -> None:
        print(f'Recognition error - request_id: {message.request_id}')
        print(f'Error message: {message.message}')
        # Stop recognition
        self.agent.stop_recognition()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if 'text' in sentence:
            text = sentence['text']
            # Call agent's text processing method
            self.agent.on_text_recognized(text)

            # Check if sentence ended
            if RecognitionResult.is_sentence_end(sentence):
                print(
                    'Sentence ended - request_id: %s, usage: %s'
                    % (result.get_request_id(), result.get_usage(sentence))
                )
                # Call speech completion callback
                self.agent.on_speech_complete(text)


# Example usage
if __name__ == '__main__':
    # Simple text processing callback
    def handle_recognized_text(text):
        print(f"Real-time recognition: {text}")


    # Create and start Speech Recognition Agent
    agent = SpeechRecognitionAgent(on_text_callback=handle_recognized_text)
    agent.start_recognition()