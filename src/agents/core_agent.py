import os
import threading
import queue
import re
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime

from .speech_recognition_agent import SpeechRecognitionAgent
from .text_analysis_agent import TextAnalysisAgent
from .command_execution_agent import CommandExecutionAgent

class CoreAgent:
    """
    会议助手的核心Agent，负责协调语音识别、文本分析等功能模块
    """
    
    def __init__(self, 
                 dashscope_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化CoreAgent
        
        Args:
            dashscope_api_key: DashScope API密钥
            openai_api_key: OpenAI API密钥
            config: 配置参数字典
        """
        # 设置API密钥环境变量
        if dashscope_api_key:
            os.environ['DASHSCOPE_API_KEY'] = dashscope_api_key
        
        # 合并默认配置
        self.config = self._merge_config(config or {})
        
        # 初始化语音识别Agent
        # 从环境变量获取唤醒关键词
        
        self.speech_recognition_agent = SpeechRecognitionAgent(
            sample_rate=self.config['sample_rate'],
            channels=self.config['channels'],
            block_size=self.config['block_size'],
            semantic_punctuation_enabled=self.config['semantic_punctuation_enabled'],
            on_text_callback=self._handle_recognized_text,
            on_speech_complete_callback=self._handle_speech_complete,
        )
        
        # 初始化文本分析Agent
        self.text_analysis_agent = TextAnalysisAgent(
            keyword_extraction_method=self.config['keyword_extraction_method'],
            max_keywords=self.config['max_keywords']
        )
        
        # 状态管理
        self.is_running = False
        self.recognized_text_buffer = []  # 存储识别的文本
        self.current_speaker = None  # 当前说话人
        self.last_text_update = datetime.now()
        
        # 结果队列和回调
        self.result_queue = queue.Queue()
        self.on_text_update_callback = None
        self.on_summary_update_callback = None
        self.on_keywords_update_callback = None
        self.on_speech_complete_callback = None
        self.on_command_executed_callback = None
        
        # 初始化指令执行Agent（不再需要传入text_analysis_agent）
        self.command_execution_agent = CommandExecutionAgent()
        
        # 唤醒关键词和指令处理
        self.wake_up_keyword = os.environ.get('WAKE_UP_KEYWORD', '小费同学')
        self.is_awake = False
        self.command_buffer = []
        self.full_meeting_text = []  # 存储完整的会议文本
        
        # 语音检测相关
        self.last_speech_time = datetime.now()
        self.speech_timeout_seconds = 2.0  # 2秒无新语音视为语音结束
        
        # 分析线程
        self.analysis_thread = None
        self.stop_analysis_thread = False
    
    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并默认配置和用户配置
        
        Args:
            user_config: 用户提供的配置
            
        Returns:
            合并后的配置
        """
        default_config = {
            'speech_model': os.environ['DASHSCOPE_ASR_MODEL_NAME'],
            'sample_rate': 16000,
            'channels': 1,
            'block_size': 3200,
            'semantic_punctuation_enabled': False,
            'keyword_extraction_method': 'llm',
            'max_keywords': 10,
            'analysis_interval_seconds': 5,  # 分析间隔时间（秒）
            'max_text_buffer_length': 10000  # 最大文本缓冲区长度
        }
        
        merged_config = default_config.copy()
        merged_config.update(user_config)
        return merged_config
    
    def start(self):
        """
        启动会议助手
        """
        if self.is_running:
            print("会议助手已经在运行中")
            return

        self.is_running = True
        self.stop_analysis_thread = False

        # 启动分析线程
        self.analysis_thread = threading.Thread(target=self._analysis_worker)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

        print("会议助手已启动")
        print(f"语音识别模型: {self.config['speech_model']}")
        print(f"关键词提取方法: {self.config['keyword_extraction_method']}")

        # 启动语音识别
        try:
            self.speech_recognition_agent.start_recognition()
        except Exception as e:
            print(f"启动语音识别时出错: {e}")
            self.stop()
            # 重新抛出异常，让调用方（如GUI）能够捕获和处理
            raise
    
    def stop(self):
        """
        停止会议助手
        """
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_analysis_thread = True
        
        # 停止语音识别
        self.speech_recognition_agent.stop_recognition()
        
        # 等待分析线程结束
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)
        
        print("会议助手已停止")
    
    def _handle_recognized_text(self, text: str):
        """
        处理识别到的文本
        
        Args:
            text: 识别到的文本
        """
        # 更新最后语音时间
        self.last_speech_time = datetime.now()
        
        # 添加到缓冲区
        self.recognized_text_buffer.append(text)
        self.full_meeting_text.append(text)
        
        # 限制缓冲区大小
        if len(' '.join(self.recognized_text_buffer)) > self.config['max_text_buffer_length']:
            # 移除最旧的部分
            while len(' '.join(self.recognized_text_buffer)) > self.config['max_text_buffer_length'] and self.recognized_text_buffer:
                self.recognized_text_buffer.pop(0)
        
        # 更新最后更新时间
        self.last_text_update = datetime.now()
        
        # 合并当前文本
        current_text = ' '.join(self.recognized_text_buffer)
        full_meeting_text = ' '.join(self.full_meeting_text)

        
        # 发送到结果队列
        result = {
            'type': 'text_update',
            'text': text,
            'full_text': current_text,
            'timestamp': datetime.now().isoformat()
        }
        self.result_queue.put(result)
        
        # 调用文本更新回调
        if self.on_text_update_callback:
            self.on_text_update_callback(text, current_text)
    
    def _handle_speech_complete(self, text: str):
        """
        处理语音完成事件
        
        Args:
            text: 完成的语音文本
        """
        # 合并当前缓冲区的完整文本

        # 检查唤醒关键词和处理指令
        self._check_wake_up_and_handle_command(text)

        full_text = ' '.join(self.recognized_text_buffer)
        
        # 调用语音完成回调
        if self.on_speech_complete_callback and full_text:
            self.on_speech_complete_callback(full_text)
        
        # 检查是否有待执行的指令
        if self.is_awake and self.command_buffer:
            command_text = ' '.join(self.command_buffer)
            self._execute_command(command_text, ' '.join(self.full_meeting_text))
            self.command_buffer = []
            self.is_awake = False  # 执行完指令后进入睡眠状态
        
        # 清空缓冲区，为新的语音做准备
        self.recognized_text_buffer = []
    
    def _analysis_worker(self):
        """
        分析线程工作函数，定期对累积的文本进行分析
        """
        import time
        
        while not self.stop_analysis_thread:
            try:
                # 检查是否有足够的文本进行分析
                if len(self.recognized_text_buffer) > 0:
                    # 检查距离上次分析的时间
                    time_since_update = (datetime.now() - self.last_text_update).total_seconds()
                    
                    # 如果距离上次更新超过指定时间，进行分析
                    if time_since_update > self.config['analysis_interval_seconds']:
                        self._perform_analysis()
                
                # 短暂睡眠避免CPU占用过高
                time.sleep(0.5)
                
            except Exception as e:
                print(f"分析线程出错: {e}")
                time.sleep(1)
    
    def _perform_analysis(self):
        """
        执行文本分析，包括关键词提取和总结生成
        """
        # 获取完整文本
        full_text = ' '.join(self.recognized_text_buffer)
        
        # 提取关键词
        keywords = self.text_analysis_agent.extract_keywords(full_text)
        
        # 生成简洁总结
        concise_summary = self.text_analysis_agent.generate_summary(full_text, "concise")
        
        # 发送到结果队列
        analysis_result = {
            'type': 'analysis_update',
            'keywords': keywords,
            'concise_summary': concise_summary,
            'timestamp': datetime.now().isoformat()
        }
        self.result_queue.put(analysis_result)
        
        # 调用回调函数
        if self.on_keywords_update_callback:
            self.on_keywords_update_callback(keywords)
        
        if self.on_summary_update_callback:
            self.on_summary_update_callback(concise_summary)
    
    def set_text_update_callback(self, callback: Callable[[str, str], None]):
        """
        设置文本更新回调函数
        
        Args:
            callback: 回调函数，接收(new_text, full_text)参数
        """
        self.on_text_update_callback = callback
    
    def set_summary_update_callback(self, callback: Callable[[str], None]):
        """
        设置总结更新回调函数
        
        Args:
            callback: 回调函数，接收summary参数
        """
        self.on_summary_update_callback = callback
    
    def set_keywords_update_callback(self, callback: Callable[[List[tuple]], None]):
        """
        设置关键词更新回调函数
        
        Args:
            callback: 回调函数，接收keywords参数
        """
        self.on_keywords_update_callback = callback
    
    def set_speech_complete_callback(self, callback: Callable[[str], None]):
        """
        设置语音完成回调函数
        
        Args:
            callback: 回调函数，接收speech_text参数
        """
        self.on_speech_complete_callback = callback
    
    def set_command_executed_callback(self, callback: Callable[[str], None]):
        """
        设置指令执行回调函数
        
        Args:
            callback: 回调函数，接收command_result参数
        """
        self.on_command_executed_callback = callback
    
    def _check_wake_up_and_handle_command(self, text: str):
        """
        检查唤醒关键词并处理指令
        
        Args:
            text: 当前识别的文本
            full_meeting_text: 完整的会议文本
        """
        # 检查是否包含唤醒关键词
        if self.wake_up_keyword in text:
            self.is_awake = True
            print(f"已唤醒！开始接收指令...")
            # 提取唤醒关键词后的内容作为指令的开始部分
            keyword_index = text.find(self.wake_up_keyword)
            if keyword_index >= 0:
                command_part = text[keyword_index + len(self.wake_up_keyword):].strip()
                if command_part:
                    self.command_buffer.append(command_part)
        # 如果已经被唤醒，将文本添加到指令缓冲区
        elif self.is_awake:
            self.command_buffer.append(text)
    
    def _execute_command(self, command_text: str, full_meeting_text: str):
        """
        执行指令（异步模式）
        
        Args:
            command_text: 指令文本
            full_meeting_text: 完整的会议文本
        """
        print(f"执行指令: {command_text}")
        
        # 清理指令文本，去除重复内容和不完整片段
        def _clean_command_text(text):
            if not text:
                return ""
            # 去除重复的前缀
            lines = text.split('\n')
            unique_lines = []
            seen = set()
            for line in lines:
                if line.strip() and line.strip() not in seen:
                    seen.add(line.strip())
                    unique_lines.append(line)
            cleaned_text = '\n'.join(unique_lines)
            
            # 提取完整的指令（如果有结束标记）
            if '。' in cleaned_text or '!' in cleaned_text or '?' in cleaned_text or '.' in cleaned_text or '！' in cleaned_text or '？' in cleaned_text:
                last_period = max(cleaned_text.rfind('。'), cleaned_text.rfind('!'), cleaned_text.rfind('?'), 
                                cleaned_text.rfind('.'), cleaned_text.rfind('！'), cleaned_text.rfind('？'))
                if last_period >= 0:
                    cleaned_text = cleaned_text[:last_period + 1]
            
            return cleaned_text.strip()
        
        # 清理指令文本
        command_text = _clean_command_text(command_text)
        
        if not command_text or re.match(r'^[，。！？；：,.;:!?]+$', command_text):
            print("无效指令，已忽略")
            return
        
        # 定义指令执行完成后的回调函数
        def command_completion_callback(result: str):
            # 确保结果不为空
            if not result or result.strip() == "":
                result = "指令执行成功，但未返回任何结果"
            
            # 发送到结果队列
            command_result = {
                'type': 'command_executed',
                'command': command_text,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            self.result_queue.put(command_result)
            
            # 调用指令执行回调，确保在主线程中更新GUI
            print(f"调用指令执行回调，结果长度: {len(result)}")
            if self.on_command_executed_callback:
                self.on_command_executed_callback(result)
        
        # 使用指令执行Agent异步处理指令
        try:
            initial_result = self.command_execution_agent.process_command(
                command_text, full_meeting_text, command_completion_callback
            )
            
            # 发送初始确认信息到结果队列
            initial_command_result = {
                'type': 'command_submitted',
                'command': command_text,
                'status': 'submitted',
                'message': initial_result,
                'timestamp': datetime.now().isoformat()
            }
            self.result_queue.put(initial_command_result)
            
            print(f"指令已提交执行，初始结果: {initial_result}")
        except Exception as e:
            error_message = f"指令执行失败: {str(e)}"
            print(error_message)
            # 直接调用回调显示错误信息
            if self.on_command_executed_callback:
                self.on_command_executed_callback(error_message)
    

    
    def get_current_text(self) -> str:
        """
        获取当前识别的完整文本
        
        Returns:
            当前识别的完整文本
        """
        return ' '.join(self.recognized_text_buffer)
    
    def get_current_analysis(self) -> Dict[str, Any]:
        """
        获取当前的分析结果
        
        Returns:
            包含关键词和总结的分析结果
        """
        full_text = ' '.join(self.recognized_text_buffer)
        
        if not full_text.strip():
            return {
                'keywords': [],
                'comprehensive_summary': "",
                'concise_summary': ""
            }
        
        keywords = self.text_analysis_agent.extract_keywords(full_text)
        comprehensive_summary = self.text_analysis_agent.generate_summary(full_text, "comprehensive")
        concise_summary = self.text_analysis_agent.generate_summary(full_text, "concise")
        
        return {
            'keywords': keywords,
            'comprehensive_summary': comprehensive_summary,
            'concise_summary': concise_summary
        }
    
    def execute_command_directly(self, command_text: str) -> str:
        """
        直接执行指令（用于GUI按钮触发），异步执行
        
        Args:
            command_text: 指令文本
        
        Returns:
            指令提交确认信息
        """
        full_meeting_text = ' '.join(self.full_meeting_text)
        
        # 定义指令执行完成后的回调函数
        def command_completion_callback(result: str):
            # 调用回调
            if self.on_command_executed_callback:
                self.on_command_executed_callback(result)
        
        # 异步执行指令
        initial_result = self.command_execution_agent.process_command(
            command_text, full_meeting_text, command_completion_callback
        )
        
        return initial_result
    
    def set_current_speaker(self, speaker: str):
        """
        设置当前说话人
        
        Args:
            speaker: 说话人标识，如'A', 'B', '主持人'等
        """
        self.current_speaker = speaker
    
    def get_next_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        获取下一个结果，用于流式处理
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            结果字典，如果超时则返回None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

