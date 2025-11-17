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
    语音识别Agent，负责将麦克风输入的语音实时转录为文本
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
        初始化语音识别Agent
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            block_size: 缓冲区大小
            format_pcm: 音频格式
            semantic_punctuation_enabled: 是否启用语义标点
            on_text_callback: 文本识别回调函数
            on_speech_complete_callback: 语音完成回调函数
            keywords: 需要增强识别的关键词列表
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
        # 从环境变量获取唤醒关键词
        if 'WAKE_UP_KEYWORD' in os.environ:
            self.keywords.append(os.environ['WAKE_UP_KEYWORD'])

        # 热词表相关属性
        self.vocabulary_service = None
        self.vocabulary_id = None

        self.mic = None
        self.stream = None
        self.recognition = None
        self.is_recording = False

        # 初始化API密钥
        self._init_api_key()

        self.mic = None
        self.stream = None
        self.recognition = None
        self.is_recording = False

        # 初始化API密钥
        self._init_api_key()

    def _init_api_key(self):
        """
        初始化DashScope API密钥
        """
        if 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            print("警告: 未找到DASHSCOPE_API_KEY环境变量，请确保正确设置")

    def _create_vocabulary(self):
        """
        创建热词表
        """
        if not self.keywords:
            return None

        try:
            self.vocabulary_service = VocabularyService()
            # 准备热词数据
            vocabulary_data = [
                {"text": keyword, "weight": 4, "lang": "zh"}
                for keyword in self.keywords
            ]

            # 创建热词表
            print(f"正在创建热词表，关键词: {self.keywords}")
            self.vocabulary_id = self.vocabulary_service.create_vocabulary(
                prefix="meeting",
                target_model=self.model,
                vocabulary=vocabulary_data
            )
            print(f"热词表创建成功，ID: {self.vocabulary_id}")
            return self.vocabulary_id
        except Exception as e:
            print(f"创建热词表失败: {e}")
            return None

    def _delete_vocabulary(self):
        """
        删除热词表
        """
        if self.vocabulary_service and self.vocabulary_id:
            try:
                print(f"正在删除热词表: {self.vocabulary_id}")
                self.vocabulary_service.delete_vocabulary(self.vocabulary_id)
                print("热词表已删除")
            except Exception as e:
                print(f"删除热词表失败: {e}")
            finally:
                self.vocabulary_id = None
                self.vocabulary_service = None

    def start_recognition(self):
        """
        开始语音识别
        """
        if self.is_recording:
            print("语音识别已经在运行中")
            return

        # 创建回调处理器
        callback = CustomRecognitionCallback(self)

        # 创建热词表
        vocabulary_id = self._create_vocabulary()

        # 初始化识别服务
        self.recognition = Recognition(
            model=self.model,
            format=self.format_pcm,
            sample_rate=self.sample_rate,
            semantic_punctuation_enabled=self.semantic_punctuation_enabled,
            callback=callback,
            vocabulary_id=vocabulary_id,
            language_hints=['zh', 'en']
        )

        # 启动识别
        self.recognition.start()
        self.is_recording = True

        # 此处不再设置signal处理器，由调用方负责停止
        print("语音识别已启动")

        # 开始音频流传输
        self._start_audio_stream()

    def _start_audio_stream(self):
        """
        开始音频流传输
        """
        try:
            while self.is_recording:
                if self.stream:
                    data = self.stream.read(self.block_size, exception_on_overflow=False)
                    self.recognition.send_audio_frame(data)
                else:
                    break
        except Exception as e:
            print(f"音频流传输错误: {e}")
            self.stop_recognition()

    def stop_recognition(self):
        """
        停止语音识别
        """
        if not self.is_recording:
            return

        self.is_recording = False

        # 停止识别服务
        if self.recognition:
            self.recognition.stop()
            print('语音识别已停止')
            print(
                '[指标] requestId: {}, 首包延迟ms: {}, 末包延迟ms: {}'
                .format(
                    self.recognition.get_last_request_id(),
                    self.recognition.get_first_package_delay(),
                    self.recognition.get_last_package_delay(),
                )
            )

        # 清理资源
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.mic:
            self.mic.terminate()
            self.mic = None

        # 删除热词表
        self._delete_vocabulary()

    def _signal_handler(self, sig, frame):
        """
        信号处理器，处理Ctrl+C中断
        """
        print('\n检测到中断信号，正在停止语音识别...')
        self.stop_recognition()

    def on_text_recognized(self, text: str):
        """
        当识别到文本时的处理方法
        
        Args:
            text: 识别到的文本
        """
        if self.on_text_callback:
            self.on_text_callback(text)
        else:
            print(f'识别文本: {text}')

    def on_speech_complete(self, text: str):
        """
        当语音完成时的处理方法
        
        Args:
            text: 完整的语音文本
        """
        if self.on_speech_complete_callback:
            self.on_speech_complete_callback(text)
        else:
            print(f'语音完成: {text}')


class CustomRecognitionCallback(RecognitionCallback):
    """
    自定义的识别回调类，用于处理识别事件
    """

    def __init__(self, agent: SpeechRecognitionAgent):
        self.agent = agent

    def on_open(self) -> None:
        print('识别连接已建立')
        # 初始化麦克风
        self.agent.mic = pyaudio.PyAudio()
        self.agent.stream = self.agent.mic.open(
            format=pyaudio.paInt16,
            channels=self.agent.channels,
            rate=self.agent.sample_rate,
            input=True
        )

    def on_close(self) -> None:
        print('识别连接已关闭')
        # 清理麦克风资源
        if self.agent.stream:
            self.agent.stream.stop_stream()
            self.agent.stream.close()
            self.agent.stream = None

        if self.agent.mic:
            self.agent.mic.terminate()
            self.agent.mic = None

    def on_complete(self) -> None:
        print('识别任务完成')

    def on_error(self, message) -> None:
        print(f'识别错误 - request_id: {message.request_id}')
        print(f'错误信息: {message.message}')
        # 停止识别
        self.agent.stop_recognition()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if 'text' in sentence:
            text = sentence['text']
            # 调用代理的文本处理方法
            self.agent.on_text_recognized(text)

            # 检查句子是否结束
            if RecognitionResult.is_sentence_end(sentence):
                print(
                    '句子结束 - request_id: %s, 用量: %s'
                    % (result.get_request_id(), result.get_usage(sentence))
                )
                # 调用语音完成回调
                self.agent.on_speech_complete(text)


# 示例用法
if __name__ == '__main__':
    # 简单的文本处理回调
    def handle_recognized_text(text):
        print(f"实时识别: {text}")


    # 创建并启动语音识别Agent
    agent = SpeechRecognitionAgent(on_text_callback=handle_recognized_text)
    agent.start_recognition()
