import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import sys
import threading
from datetime import datetime

# 添加src目录到Python路径
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.agents.core_agent import CoreAgent

class MeetingAssistantGUI:
    """
    会议助手GUI界面，包含启动按钮、会议记录显示区域和会议总结显示区域
    """
    
    def __init__(self, root):
        """
        初始化GUI
        
        Args:
            root: tkinter的根窗口
        """
        self.root = root
        self.root.title("会议助手 - 实时语音记录与总结")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        
        # 设置窗口图标（如果有）
        # self.root.iconbitmap("path_to_icon.ico")
        
        # 设置中文字体
        self._setup_fonts()
        
        # 初始化CoreAgent
        self.core_agent = None
        self.is_running = False
        self.current_speech = ""  # 存储当前语音内容
        
        # 创建GUI组件
        self._create_widgets()
        
        # 配置样式
        self._configure_styles()
        
        # 配置回调函数
        self._configure_callbacks()
    
    def _setup_fonts(self):
        """
        设置中文字体
        """
        # 尝试使用系统中的中文字体
        self.font_family = "SimHei"
        self.normal_font = (self.font_family, 10)
        self.header_font = (self.font_family, 12, "bold")
        self.button_font = (self.font_family, 10)
    
    def _create_widgets(self):
        """
        创建GUI组件
        """
        # 顶部控制区域
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        # 启动/停止按钮
        self.start_stop_button = ttk.Button(
            control_frame,
            text="开始记录",
            command=self.toggle_recording,
            width=20,
            style="Accent.TButton"
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        self.status_label = ttk.Label(
            control_frame,
            textvariable=self.status_var,
            font=self.normal_font,
            foreground="green"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # 时间标签
        self.time_var = tk.StringVar()
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.time_label = ttk.Label(
            control_frame,
            textvariable=self.time_var,
            font=self.normal_font
        )
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        # 更新时间
        self._update_time()
        
        # 第一部分 - 当前语音显示区域
        current_speech_frame = ttk.LabelFrame(self.root, 
                                             text="当前语音", 
                                             padding="10",
                                             style="CurrentSpeech.TLabelframe")
        current_speech_frame.pack(fill=tk.BOTH, expand=False, side=tk.TOP, padx=10, pady=5)
        
        self.current_speech_text = scrolledtext.ScrolledText(
            current_speech_frame,
            wrap=tk.WORD,
            font=(self.font_family, 11),  # 稍大一点的字体
            height=5,  # 限制高度
            state=tk.DISABLED,
            background="#ffffff",
            bd=1,
            relief=tk.SUNKEN,
            highlightbackground="#0066cc",
            highlightthickness=0
        )
        self.current_speech_text.pack(fill=tk.BOTH, expand=True)
        
        # 第二部分 - 会议发言记录显示区域（缩短）
        speech_records_frame = ttk.LabelFrame(self.root, 
                                             text="会议发言记录", 
                                             padding="10",
                                             style="SpeechRecords.TLabelframe")
        speech_records_frame.pack(fill=tk.BOTH, expand=False, side=tk.TOP, padx=10, pady=5, ipady=0)
        
        self.speech_records_text = scrolledtext.ScrolledText(
            speech_records_frame,
            wrap=tk.WORD,
            font=self.normal_font,
            height=10,  # 限制高度
            state=tk.DISABLED,
            background="#ffffff",
            bd=1,
            relief=tk.SUNKEN,
            highlightbackground="#cccccc",
            highlightthickness=0
        )
        self.speech_records_text.pack(fill=tk.BOTH, expand=True)
        
        # 第三部分 - 指令执行结果反馈区域（拉长）
        command_result_frame = ttk.LabelFrame(self.root, 
                                      text="指令执行结果", 
                                      padding="10",
                                      style="CommandResult.TLabelframe")
        command_result_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=10, pady=5, ipady=10)
        
        # 指令输入区域
        command_input_frame = ttk.Frame(command_result_frame)
        command_input_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        ttk.Label(command_input_frame, text="输入指令:", font=self.normal_font).pack(side=tk.LEFT, padx=5)
        
        self.command_entry = ttk.Entry(command_input_frame, font=self.normal_font)
        self.command_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5)
        self.command_entry.bind("<Return>", self.on_command_entry_enter)
        
        self.execute_command_button = ttk.Button(
            command_input_frame,
            text="执行",
            command=self.execute_command,
            style="Accent.TButton"
        )
        self.execute_command_button.pack(side=tk.LEFT, padx=5)
        
        # 快捷指令按钮
        shortcut_frame = ttk.Frame(command_result_frame)
        shortcut_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        ttk.Label(shortcut_frame, text="快捷指令:", font=self.normal_font).pack(side=tk.LEFT, padx=5)
        
        shortcuts = [
            ("总结会议", "总结会议内容"),
            ("提取关键词", "提取核心关键词"),
            ("查找预算", "查找项目预算信息"),
            ("时间安排", "总结时间节点和计划")
        ]
        
        for text, command in shortcuts:
            btn = ttk.Button(
                shortcut_frame,
                text=text,
                command=lambda cmd=command: self.execute_shortcut_command(cmd),
                style="TButton"
            )
            btn.pack(side=tk.LEFT, padx=3)
        
        # 指令结果显示区域
        self.command_result_text = scrolledtext.ScrolledText(
            command_result_frame,
            wrap=tk.WORD,
            font=self.normal_font,
            state=tk.DISABLED,
            background="#ffffff",
            bd=1,
            relief=tk.SUNKEN,
            highlightbackground="#0066cc",
            highlightthickness=0
        )
        self.command_result_text.pack(fill=tk.BOTH, expand=True)
    
    def _configure_styles(self):
        """
        配置ttk样式
        """
        style = ttk.Style()
        
        # 按钮样式
        style.configure("Accent.TButton", font=self.button_font, padding=5)
        
        # 框架样式 - 添加视觉分隔
        style.configure("TFrame", background="#f0f0f0")
        
        # 标签框架样式
        style.configure("TLabelframe", 
                        background="#f0f0f0",
                        font=self.header_font,
                        labeloutside=False)
        style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#333333")
        
        # 为不同模块定义不同的标签框架样式
        style.configure("CurrentSpeech.TLabelframe", background="#e6f3ff")
        style.configure("CurrentSpeech.TLabelframe.Label", 
                        background="#e6f3ff", 
                        foreground="#0066cc",
                        font=(self.font_family, 12, "bold"))
        
        style.configure("SpeechRecords.TLabelframe", background="#f5f5f5")
        style.configure("SpeechRecords.TLabelframe.Label", 
                        background="#f5f5f5", 
                        foreground="#333333",
                        font=(self.font_family, 12, "bold"))
        
        style.configure("CommandResult.TLabelframe", background="#f0f8ff")
        style.configure("CommandResult.TLabelframe.Label", 
                        background="#f0f8ff", 
                        foreground="#0066cc",
                        font=(self.font_family, 12, "bold"))
    
    def _update_time(self):
        """
        更新时间标签
        """
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._update_time)
    
    def _configure_callbacks(self):
        """
        配置回调函数
        """
        # 窗口关闭时的处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def toggle_recording(self):
        """
        切换录音状态
        """
        if not self.is_running:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """
        开始录音和处理
        """
        try:
            # 加载环境变量中的API密钥
            from dotenv import load_dotenv
            load_dotenv()
            
            dashscope_key = os.getenv('DASHSCOPE_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            # 创建CoreAgent实例
            config = {
                'analysis_interval_seconds': 5,  # 每5秒分析一次
                'keyword_extraction_method': 'jieba'  # 使用jieba提取关键词
            }
            
            self.core_agent = CoreAgent(
                dashscope_api_key=dashscope_key,
                openai_api_key=openai_key,
                config=config
            )
            
            # 设置回调函数
            self.core_agent.set_text_update_callback(self.on_text_update)
            # 将原来的总结回调改为指令执行结果回调
            if hasattr(self.core_agent, 'set_command_executed_callback'):
                self.core_agent.set_command_executed_callback(self.on_command_executed)
            # 添加语音完成回调
            if hasattr(self.core_agent, 'set_speech_complete_callback'):
                self.core_agent.set_speech_complete_callback(self.on_speech_complete)
            
            # 在单独的线程中启动CoreAgent
            self.recording_thread = threading.Thread(target=self._run_core_agent)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # 更新UI状态
            self.is_running = True
            self.start_stop_button.config(text="停止记录")
            self.status_var.set("正在记录...")
            self.status_label.config(foreground="red")
            
            self._append_to_speech_records("[系统] 会议记录已开始，请开始讲话...")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动会议记录时出错: {str(e)}")
            self._append_to_speech_records(f"[系统错误] {str(e)}")
    
    def _run_core_agent(self):
        """
        在单独的线程中运行CoreAgent
        """
        try:
            self.core_agent.start()
        except Exception as e:
            error_msg = str(e)
            # 捕获并处理特定错误
            if "signal only works" in error_msg:
                error_msg = "语音识别启动失败：主线程错误"
            self.root.after(0, lambda: messagebox.showerror("错误", f"CoreAgent运行出错: {error_msg}"))
    
    def stop_recording(self):
        """
        停止录音和处理
        """
        try:
            if self.core_agent:
                self.core_agent.stop()
                
            # 更新UI状态
            self.is_running = False
            self.start_stop_button.config(text="开始记录")
            self.status_var.set("已停止")
            self.status_label.config(foreground="green")
            
            # 保存当前语音到会议发言记录
            self.current_speech = self.current_speech_text.get(1.0, tk.END).strip()
            if self.current_speech and not self.current_speech.startswith("[系统]"):
                self._append_to_speech_records(self.current_speech)
                self._update_current_speech("")
                
            self._append_to_speech_records("[系统] 会议记录已停止")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止会议记录时出错: {str(e)}")
    
    def on_speech_complete(self, speech_text):
        """
        语音完成回调函数，当一段语音结束时被调用
        
        Args:
            speech_text: 完成的语音文本
        """
        # 将当前语音保存到会议发言记录
        self.root.after(0, lambda: self._save_current_speech())
    
    def _save_current_speech(self):
        """
        保存当前语音到会议发言记录，并清空当前语音区域
        """
        speech_text = self.current_speech_text.get(1.0, tk.END).strip()
        if speech_text and not speech_text.startswith("[系统]"):
            self._append_to_speech_records(speech_text)
            self._update_current_speech("")
    
    def on_text_update(self, new_text, full_text):
        """
        文本更新回调函数
        
        Args:
            new_text: 新识别的文本
            full_text: 完整的文本
        """
        # 在主线程中更新UI
        self.root.after(0, lambda: self._update_current_speech(new_text))
    
    def on_command_executed(self, result):
        """
        指令执行结果回调函数
        
        Args:
            result: 指令执行结果
        """
        print(f"GUI收到指令执行结果，长度: {len(result)}")
        # 在主线程中更新UI
        self.root.after(0, lambda: self._update_command_result(result))
    
    def _update_current_speech(self, text):
        """
        更新当前语音显示区域
        
        Args:
            text: 要显示的文本
        """
        self.current_speech_text.config(state=tk.NORMAL)
        self.current_speech_text.delete(1.0, tk.END)  # 清空现有内容
        self.current_speech_text.insert(tk.END, text)
        # 如果正在录音，添加一个视觉指示器（闪烁光标）
        if self.is_running:
            self.current_speech_text.mark_set(tk.INSERT, tk.END)
        self.current_speech_text.config(state=tk.DISABLED)
    
    def _append_to_speech_records(self, text):
        """
        添加文本到会议发言记录区域
        
        Args:
            text: 要添加的文本
        """
        self.speech_records_text.config(state=tk.NORMAL)
        
        # 为系统消息和普通发言设置不同的样式
        if text.startswith("[系统]") or text.startswith("[系统错误]"):
            # 系统消息使用不同的颜色
            self.speech_records_text.insert(tk.END, f"{text}\n", "system_message")
        else:
            # 普通发言使用默认样式，添加时间戳
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.speech_records_text.insert(tk.END, f"[{timestamp}] {text}\n")
        
        # 设置系统消息的标签样式
        self.speech_records_text.tag_configure("system_message", foreground="#666666", font=(self.font_family, 10, "italic"))
        
        self.speech_records_text.see(tk.END)  # 自动滚动到底部
        self.speech_records_text.config(state=tk.DISABLED)
    
    def _update_command_result(self, result):
        """
        更新指令执行结果区域
        
        Args:
            result: 指令执行结果
        """
        print("开始更新指令执行结果显示区域")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.command_result_text.config(state=tk.NORMAL)
        
        # 添加分隔线以增强可读性
        separator = "=" * 60 + "\n"
        # 在结果前添加时间戳和标记
        formatted_result = f"{separator}[{timestamp}] 指令结果:\n\n{result}\n\n"
        
        # 插入到开头，最新的结果在前面
        self.command_result_text.insert(1.0, formatted_result)
        
        # 限制结果显示的行数，避免内容过长
        lines = self.command_result_text.get(1.0, tk.END).split('\n')
        if len(lines) > 150:  # 保留最近的150行
            self.command_result_text.delete(1.0, f"{len(lines) - 150}.0")
        
        # 确保显示区域可编辑后再滚动
        self.command_result_text.see(1.0)  # 滚动到顶部（最新结果）
        self.command_result_text.config(state=tk.DISABLED)
        print("指令执行结果显示区域更新完成")
    
    def on_command_entry_enter(self, event):
        """
        处理指令输入框的回车键事件
        
        Args:
            event: 事件对象
        """
        self.execute_command()
    
    def execute_command(self):
        """
        执行用户输入的指令
        """
        command_text = self.command_entry.get().strip()
        if command_text:
            try:
                if self.core_agent and self.is_running:
                    self._append_to_speech_records(f"[系统] 执行指令: {command_text}")
                    # 直接执行指令
                    result = self.core_agent.execute_command_directly(command_text)
                    # 更新结果显示
                    self._update_command_result(result)
                else:
                    messagebox.showwarning("警告", "请先开始录音")
            except Exception as e:
                messagebox.showerror("错误", f"执行指令时出错: {str(e)}")
            finally:
                # 清空输入框
                self.command_entry.delete(0, tk.END)
    
    def execute_shortcut_command(self, command):
        """
        执行快捷指令
        
        Args:
            command: 快捷指令文本
        """
        self.command_entry.delete(0, tk.END)
        self.command_entry.insert(0, command)
        self.execute_command()
    
    def on_closing(self):
        """
        窗口关闭时的处理
        """
        if self.is_running:
            if messagebox.askyesno("确认", "正在录音中，确定要退出吗？"):
                self.stop_recording()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """
    主函数
    """
    import sys
    
    # 确保中文显示正常
    if sys.platform == 'win32':
        # Windows系统已经在_setup_fonts中设置了字体
        pass
    
    root = tk.Tk()
    app = MeetingAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()