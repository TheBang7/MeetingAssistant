import os
import threading
from typing import Optional, Dict, Any, Callable
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

class CommandExecutionAgent:
    """
    指令执行Agent，负责处理用户指令并返回执行结果
    支持异步执行，通过线程方式避免阻塞主线程
    """
    
    def __init__(self):
        """
        初始化指令执行Agent，直接初始化LLM
        """
        # 从环境变量获取配置
        self.llm_model = os.getenv("DASHSCOPE_LLM_MODEL_NAME", "deepseek-r1")
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.dashscope_llm_base_url = os.getenv("DASHSCOPE_LLM_BASE_URL",
                                                "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 初始化LLM
        if self.dashscope_api_key:
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.dashscope_api_key,
                base_url=self.dashscope_llm_base_url
            )
        else:
            self.llm = None
            print("警告: 未找到DASHSCOPE_API_KEY环境变量，LLM功能将不可用")
        
        # 线程管理
        self.execution_threads = []
    
    def process_command(self, command_text: str, full_meeting_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """
        处理具体的指令，支持同步和异步模式
        
        Args:
            command_text: 指令文本
            full_meeting_text: 完整的会议文本
            callback: 可选的回调函数，当提供时异步执行指令
        
        Returns:
            同步模式下返回指令执行结果，异步模式下返回确认信息
        """
        print(f"[CommandExecutionAgent] 收到指令: '{command_text}'")
        # 如果提供了回调函数，则异步执行
        if callback:
            # 创建并启动新线程执行指令
            thread = threading.Thread(
                target=self._async_process_command,
                args=(command_text, full_meeting_text, callback),
                daemon=True
            )
            self.execution_threads.append(thread)
            thread.start()
            
            # 清理已完成的线程
            self._cleanup_finished_threads()
            
            # 返回异步执行确认
            confirm_msg = f"指令 '{command_text}' 已提交，正在异步处理中..."
            print(f"[CommandExecutionAgent] {confirm_msg}")
            return confirm_msg
        else:
            # 同步模式，直接返回结果
            return self._process_with_llm(command_text, full_meeting_text)
    
    def _async_process_command(self, command_text: str, full_meeting_text: str, callback: Callable[[str], None]):
        """
        异步处理指令的内部方法
        
        Args:
            command_text: 指令文本
            full_meeting_text: 完整的会议文本
            callback: 处理完成后的回调函数
        """
        try:
            print(f"[CommandExecutionAgent] 开始处理指令: '{command_text}'")
            # 执行指令
            result = self._process_with_llm(command_text, full_meeting_text)
            print(f"[CommandExecutionAgent] 指令处理完成，结果长度: {len(result)} 字符")
            # 确保回调函数被调用，并且结果不为空
            if result and callback:
                # 调用回调函数返回结果
                callback(result)
            else:
                default_result = "指令已执行，但未返回具体结果"
                print(f"[CommandExecutionAgent] 结果为空或回调不可用，使用默认结果")
                if callback:
                    callback(default_result)
        except Exception as e:
            error_result = f"异步执行指令时出错: {str(e)}"
            print(f"[CommandExecutionAgent] {error_result}")
            # 即使出错也调用回调函数通知调用方
            if callback:
                callback(error_result)
    
    def _cleanup_finished_threads(self):
        """
        清理已完成的线程，避免内存泄漏
        """
        alive_threads = [thread for thread in self.execution_threads if thread.is_alive()]
        if len(self.execution_threads) > len(alive_threads):
            print(f"[CommandExecutionAgent] 清理了 {len(self.execution_threads) - len(alive_threads)} 个已完成的线程")
        self.execution_threads = alive_threads
    
    def _process_with_llm(self, command: str, context: str) -> str:
        """
        使用LLM处理指令
        
        Args:
            command: 指令
            context: 上下文信息
        
        Returns:
            处理结果
        """
        print(f"[CommandExecutionAgent] 调用LLM处理指令")
        try:
            # 构建更详细的提示，指导LLM如何处理不同类型的指令
            prompt_template = """
            请根据以下会议文本，回答用户的问题或执行用户的指令。
            
            会议文本：
            {context}
            
            用户指令：
            {command}
            
            处理说明：
            1. 如果是总结类指令（如总结、概述、回顾），请提供完整且准确的会议内容总结
            2. 如果是查找特定信息的指令（如预算、费用、时间、日期等），请从会议文本中提取相关信息并整理
            3. 如果是提取关键词的指令，请识别会议中的核心概念和重要术语
            4. 对于其他类型的指令，请根据会议内容提供最相关的回答
            5. 请保持回答简洁明了，直接给出答案，不要添加额外的说明
            
            请直接给出答案：
            """
            
            # 直接使用自身初始化的LLM
            if self.llm:
                chat_prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = chat_prompt | self.llm | StrOutputParser()
                result = chain.invoke({"context": context, "command": command})
                print(f"[CommandExecutionAgent] LLM返回结果: {result}")
                return result
            else:
                error_message = f"无法执行指令：{command}（LLM未初始化，请确保环境变量DASHSCOPE_API_KEY已正确设置）"
                print(f"[CommandExecutionAgent] {error_message}")
                return error_message
        except Exception as e:
            error_message = f"执行指令时出错: {str(e)}"
            print(f"[CommandExecutionAgent] 执行LLM指令时出错: {e}")
            return error_message