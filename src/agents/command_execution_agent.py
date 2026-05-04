import os
import threading
from typing import Optional, Callable

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class CommandExecutionAgent:
    """
    Command Execution Agent, responsible for processing user commands and returning execution results
    Supports asynchronous execution using threads to avoid blocking the main thread
    """

    def __init__(self):
        """
        Initialize Command Execution Agent and directly initialize LLM
        """
        # Get configuration from environment variables
        self.llm_model = os.getenv("DASHSCOPE_LLM_MODEL_NAME", "deepseek-r1")
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.dashscope_llm_base_url = os.getenv("DASHSCOPE_LLM_BASE_URL",
                                                "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # Initialize LLM
        if self.dashscope_api_key:
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.dashscope_api_key,
                base_url=self.dashscope_llm_base_url
            )
        else:
            self.llm = None
            print("Warning: DASHSCOPE_API_KEY environment variable not found, LLM functionality will be unavailable")

        # Thread management
        self.execution_threads = []

    def process_command(self, command_text: str, full_meeting_text: str,
                        callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Process specific commands, supporting both synchronous and asynchronous modes
        
        Args:
            command_text: Command text
            full_meeting_text: Complete meeting text
            callback: Optional callback function, when provided, execute command asynchronously
        
        Returns:
            Command execution result in synchronous mode, confirmation message in asynchronous mode
        """
        print(f"[CommandExecutionAgent] Received command: '{command_text}'")
        # If callback function is provided, execute asynchronously
        if callback:
            # Create and start new thread to execute command
            thread = threading.Thread(
                target=self._async_process_command,
                args=(command_text, full_meeting_text, callback),
                daemon=True
            )
            self.execution_threads.append(thread)
            thread.start()

            # Clean up finished threads
            self._cleanup_finished_threads()

            # Return asynchronous execution confirmation
            confirm_msg = f"Command '{command_text}' has been submitted, processing asynchronously..."
            print(f"[CommandExecutionAgent] {confirm_msg}")
            return confirm_msg
        else:
            # Synchronous mode, directly return result
            return self._process_with_llm(command_text, full_meeting_text)

    def _async_process_command(self, command_text: str, full_meeting_text: str, callback: Callable[[str], None]):
        """
        Internal method for asynchronous command processing
        
        Args:
            command_text: Command text
            full_meeting_text: Complete meeting text
            callback: Callback function after processing is complete
        """
        try:
            print(f"[CommandExecutionAgent] Started processing command: '{command_text}'")
            # Execute command
            result = self._process_with_llm(command_text, full_meeting_text)
            print(f"[CommandExecutionAgent] Command processing completed, result length: {len(result)} characters")
            # Ensure callback function is called and result is not empty
            if result and callback:
                # Call callback function to return result
                callback(result)
            else:
                default_result = "Command executed, but no specific result returned"
                print(f"[CommandExecutionAgent] Result is empty or callback is unavailable, using default result")
                if callback:
                    callback(default_result)
        except Exception as e:
            error_result = f"Error executing command asynchronously: {str(e)}"
            print(f"[CommandExecutionAgent] {error_result}")
            # Call callback function to notify caller even if error occurs
            if callback:
                callback(error_result)

    def _cleanup_finished_threads(self):
        """
        Clean up finished threads to avoid memory leaks
        """
        alive_threads = [thread for thread in self.execution_threads if thread.is_alive()]
        if len(self.execution_threads) > len(alive_threads):
            print(f"[CommandExecutionAgent] Cleaned up {len(self.execution_threads) - len(alive_threads)} finished threads")
        self.execution_threads = alive_threads

    def _process_with_llm(self, command: str, context: str) -> str:
        """
        Process commands using LLM
        
        Args:
            command: Command
            context: Context information
        
        Returns:
            Processing result
        """
        print(f"[CommandExecutionAgent] Calling LLM to process command")
        try:
            # Build more detailed prompt to guide LLM on how to handle different types of commands
            prompt_template = """
            Please answer the user's question or execute the user's command based on the following meeting text.
            
            Meeting text:
            {context}
            
            User command:
            {command}
            
            Processing instructions:
            1. If it's a summary type command (e.g., summarize, overview, review), please provide a complete and accurate summary of the meeting content
            2. If it's a command to find specific information (e.g., budget, cost, time, date, etc.), please extract relevant information from the meeting text and organize it
            3. If it's a command to extract keywords, please identify the core concepts and important terms in the meeting
            4. For other types of commands, please provide the most relevant answer based on the meeting content
            5. Please keep your answer concise and clear, give the answer directly without adding extra explanations
            
            Please give the answer directly:
            """

            # Directly use the LLM initialized by itself
            if self.llm:
                chat_prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = chat_prompt | self.llm | StrOutputParser()
                result = chain.invoke({"context": context, "command": command})
                print(f"[CommandExecutionAgent] LLM returned result: {result}")
                return result
            else:
                error_message = f"Unable to execute command: {command} (LLM not initialized, please ensure environment variable DASHSCOPE_API_KEY is properly set)"
                print(f"[CommandExecutionAgent] {error_message}")
                return error_message
        except Exception as e:
            error_message = f"Error executing command: {str(e)}"
            print(f"[CommandExecutionAgent] Error executing LLM command: {e}")
            return error_message