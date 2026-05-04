import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import sys
import threading
from datetime import datetime

# Add src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.agents.core_agent import CoreAgent

class MeetingAssistantGUI:
    """
    Meeting Assistant GUI interface, including start button, meeting record display area, and meeting summary display area
    """
    
    def __init__(self, root):
        """
        Initialize GUI
        
        Args:
            root: tkinter root window
        """
        self.root = root
        self.root.title("Meeting Assistant - Real-time Voice Recording & Summary")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        
        # Set window icon (if available)
        # self.root.iconbitmap("path_to_icon.ico")
        
        # Set fonts
        self._setup_fonts()
        
        # Initialize CoreAgent
        self.core_agent = None
        self.is_running = False
        self.current_speech = ""  # Store current speech content
        
        # Create GUI components
        self._create_widgets()
        
        # Configure styles
        self._configure_styles()
        
        # Configure callbacks
        self._configure_callbacks()
    
    def _setup_fonts(self):
        """
        Set fonts
        """
        # Use system fonts
        self.font_family = "Arial"
        self.normal_font = (self.font_family, 10)
        self.header_font = (self.font_family, 12, "bold")
        self.button_font = (self.font_family, 10)
    
    def _create_widgets(self):
        """
        Create GUI components
        """
        # Top control area
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Start/Stop button
        self.start_stop_button = ttk.Button(
            control_frame,
            text="Start Recording",
            command=self.toggle_recording,
            width=20,
            style="Accent.TButton"
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(
            control_frame,
            textvariable=self.status_var,
            font=self.normal_font,
            foreground="green"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Time label
        self.time_var = tk.StringVar()
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.time_label = ttk.Label(
            control_frame,
            textvariable=self.time_var,
            font=self.normal_font
        )
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        # Update time
        self._update_time()
        
        # Part 1 - Current Speech Display Area
        current_speech_frame = ttk.LabelFrame(self.root, 
                                             text="Current Speech", 
                                             padding="10",
                                             style="CurrentSpeech.TLabelframe")
        current_speech_frame.pack(fill=tk.BOTH, expand=False, side=tk.TOP, padx=10, pady=5)
        
        self.current_speech_text = scrolledtext.ScrolledText(
            current_speech_frame,
            wrap=tk.WORD,
            font=(self.font_family, 11),  # Slightly larger font
            height=5,  # Limit height
            state=tk.DISABLED,
            background="#ffffff",
            bd=1,
            relief=tk.SUNKEN,
            highlightbackground="#0066cc",
            highlightthickness=0
        )
        self.current_speech_text.pack(fill=tk.BOTH, expand=True)
        
        # Part 2 - Meeting Speech Records Display Area (shortened)
        speech_records_frame = ttk.LabelFrame(self.root, 
                                             text="Meeting Speech Records", 
                                             padding="10",
                                             style="SpeechRecords.TLabelframe")
        speech_records_frame.pack(fill=tk.BOTH, expand=False, side=tk.TOP, padx=10, pady=5, ipady=0)
        
        self.speech_records_text = scrolledtext.ScrolledText(
            speech_records_frame,
            wrap=tk.WORD,
            font=self.normal_font,
            height=10,  # Limit height
            state=tk.DISABLED,
            background="#ffffff",
            bd=1,
            relief=tk.SUNKEN,
            highlightbackground="#cccccc",
            highlightthickness=0
        )
        self.speech_records_text.pack(fill=tk.BOTH, expand=True)
        
        # Part 3 - Command Execution Result Display Area (extended)
        command_result_frame = ttk.LabelFrame(self.root, 
                                      text="Command Execution Results", 
                                      padding="10",
                                      style="CommandResult.TLabelframe")
        command_result_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=10, pady=5, ipady=10)
        
        # Command input area
        command_input_frame = ttk.Frame(command_result_frame)
        command_input_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        ttk.Label(command_input_frame, text="Enter Command:", font=self.normal_font).pack(side=tk.LEFT, padx=5)
        
        self.command_entry = ttk.Entry(command_input_frame, font=self.normal_font)
        self.command_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5)
        self.command_entry.bind("<Return>", self.on_command_entry_enter)
        
        self.execute_command_button = ttk.Button(
            command_input_frame,
            text="Execute",
            command=self.execute_command,
            style="Accent.TButton"
        )
        self.execute_command_button.pack(side=tk.LEFT, padx=5)
        
        # Shortcut command buttons
        shortcut_frame = ttk.Frame(command_result_frame)
        shortcut_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        ttk.Label(shortcut_frame, text="Shortcuts:", font=self.normal_font).pack(side=tk.LEFT, padx=5)
        
        shortcuts = [
            ("Summarize", "summarize meeting content"),
            ("Keywords", "extract core keywords"),
            ("Budget", "find project budget information"),
            ("Schedule", "summarize time nodes and plans")
        ]
        
        for text, command in shortcuts:
            btn = ttk.Button(
                shortcut_frame,
                text=text,
                command=lambda cmd=command: self.execute_shortcut_command(cmd),
                style="TButton"
            )
            btn.pack(side=tk.LEFT, padx=3)
        
        # Command result display area
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
        Configure ttk styles
        """
        style = ttk.Style()
        
        # Button style
        style.configure("Accent.TButton", font=self.button_font, padding=5)
        
        # Frame style - add visual separation
        style.configure("TFrame", background="#f0f0f0")
        
        # Label frame style
        style.configure("TLabelframe", 
                        background="#f0f0f0",
                        font=self.header_font,
                        labeloutside=False)
        style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#333333")
        
        # Define different label frame styles for different modules
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
        Update time label
        """
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._update_time)
    
    def _configure_callbacks(self):
        """
        Configure callback functions
        """
        # Window closing handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def toggle_recording(self):
        """
        Toggle recording state
        """
        if not self.is_running:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """
        Start recording and processing
        """
        try:
            # Load API keys from environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            dashscope_key = os.getenv('DASHSCOPE_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            # Create CoreAgent instance
            config = {
                'analysis_interval_seconds': 5,  # Analyze every 5 seconds
                'keyword_extraction_method': 'jieba'  # Use jieba for keyword extraction
            }
            
            self.core_agent = CoreAgent(
                dashscope_api_key=dashscope_key,
                openai_api_key=openai_key,
                config=config
            )
            
            # Set callback functions
            self.core_agent.set_text_update_callback(self.on_text_update)
            # Change original summary callback to command execution result callback
            if hasattr(self.core_agent, 'set_command_executed_callback'):
                self.core_agent.set_command_executed_callback(self.on_command_executed)
            # Add speech completion callback
            if hasattr(self.core_agent, 'set_speech_complete_callback'):
                self.core_agent.set_speech_complete_callback(self.on_speech_complete)
            
            # Start CoreAgent in a separate thread
            self.recording_thread = threading.Thread(target=self._run_core_agent)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Update UI state
            self.is_running = True
            self.start_stop_button.config(text="Stop Recording")
            self.status_var.set("Recording...")
            self.status_label.config(foreground="red")
            
            self._append_to_speech_records("[System] Meeting recording started. Please start speaking...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start meeting recording: {str(e)}")
            self._append_to_speech_records(f"[System Error] {str(e)}")
    
    def _run_core_agent(self):
        """
        Run CoreAgent in a separate thread
        """
        try:
            self.core_agent.start()
        except Exception as e:
            error_msg = str(e)
            # Capture and handle specific errors
            if "signal only works" in error_msg:
                error_msg = "Speech recognition failed to start: Main thread error"
            self.root.after(0, lambda: messagebox.showerror("Error", f"CoreAgent error: {error_msg}"))
    
    def stop_recording(self):
        """
        Stop recording and processing
        """
        try:
            if self.core_agent:
                self.core_agent.stop()
                
            # Update UI state
            self.is_running = False
            self.start_stop_button.config(text="Start Recording")
            self.status_var.set("Stopped")
            self.status_label.config(foreground="green")
            
            # Save current speech to meeting records
            self.current_speech = self.current_speech_text.get(1.0, tk.END).strip()
            if self.current_speech and not self.current_speech.startswith("[System]"):
                self._append_to_speech_records(self.current_speech)
                self._update_current_speech("")
                
            self._append_to_speech_records("[System] Meeting recording stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop meeting recording: {str(e)}")
    
    def on_speech_complete(self, speech_text):
        """
        Speech completion callback function, called when a speech segment ends
        
        Args:
            speech_text: Completed speech text
        """
        # Save current speech to meeting records
        self.root.after(0, lambda: self._save_current_speech())
    
    def _save_current_speech(self):
        """
        Save current speech to meeting records and clear current speech area
        """
        speech_text = self.current_speech_text.get(1.0, tk.END).strip()
        if speech_text and not speech_text.startswith("[System]"):
            self._append_to_speech_records(speech_text)
            self._update_current_speech("")
    
    def on_text_update(self, new_text, full_text):
        """
        Text update callback function
        
        Args:
            new_text: Newly recognized text
            full_text: Complete text
        """
        # Update UI in main thread
        self.root.after(0, lambda: self._update_current_speech(new_text))
    
    def on_command_executed(self, result):
        """
        Command execution result callback function
        
        Args:
            result: Command execution result
        """
        print(f"GUI received command execution result, length: {len(result)}")
        # Update UI in main thread
        self.root.after(0, lambda: self._update_command_result(result))
    
    def _update_current_speech(self, text):
        """
        Update current speech display area
        
        Args:
            text: Text to display
        """
        self.current_speech_text.config(state=tk.NORMAL)
        self.current_speech_text.delete(1.0, tk.END)  # Clear existing content
        self.current_speech_text.insert(tk.END, text)
        # Add visual indicator if recording (blinking cursor)
        if self.is_running:
            self.current_speech_text.mark_set(tk.INSERT, tk.END)
        self.current_speech_text.config(state=tk.DISABLED)
    
    def _append_to_speech_records(self, text):
        """
        Add text to meeting speech records area
        
        Args:
            text: Text to add
        """
        self.speech_records_text.config(state=tk.NORMAL)
        
        # Set different styles for system messages and regular speech
        if text.startswith("[System]") or text.startswith("[System Error]"):
            # System messages use different color
            self.speech_records_text.insert(tk.END, f"{text}\n", "system_message")
        else:
            # Regular speech uses default style, add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.speech_records_text.insert(tk.END, f"[{timestamp}] {text}\n")
        
        # Set tag style for system messages
        self.speech_records_text.tag_configure("system_message", foreground="#666666", font=(self.font_family, 10, "italic"))
        
        self.speech_records_text.see(tk.END)  # Auto-scroll to bottom
        self.speech_records_text.config(state=tk.DISABLED)
    
    def _update_command_result(self, result):
        """
        Update command execution result area
        
        Args:
            result: Command execution result
        """
        print("Starting to update command execution result display area")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.command_result_text.config(state=tk.NORMAL)
        
        # Add separator for better readability
        separator = "=" * 60 + "\n"
        # Add timestamp and marker before result
        formatted_result = f"{separator}[{timestamp}] Command Result:\n\n{result}\n\n"
        
        # Insert at the beginning, newest result first
        self.command_result_text.insert(1.0, formatted_result)
        
        # Limit the number of lines displayed to avoid excessive content
        lines = self.command_result_text.get(1.0, tk.END).split('\n')
        if len(lines) > 150:  # Keep latest 150 lines
            self.command_result_text.delete(1.0, f"{len(lines) - 150}.0")
        
        # Ensure display area is editable before scrolling
        self.command_result_text.see(1.0)  # Scroll to top (newest result)
        self.command_result_text.config(state=tk.DISABLED)
        print("Command execution result display area updated")
    
    def on_command_entry_enter(self, event):
        """
        Handle enter key event in command input box
        
        Args:
            event: Event object
        """
        self.execute_command()
    
    def execute_command(self):
        """
        Execute user-input command
        """
        command_text = self.command_entry.get().strip()
        if command_text:
            try:
                if self.core_agent and self.is_running:
                    self._append_to_speech_records(f"[System] Executing command: {command_text}")
                    # Directly execute command
                    result = self.core_agent.execute_command_directly(command_text)
                    # Update result display
                    self._update_command_result(result)
                else:
                    messagebox.showwarning("Warning", "Please start recording first")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to execute command: {str(e)}")
            finally:
                # Clear input box
                self.command_entry.delete(0, tk.END)
    
    def execute_shortcut_command(self, command):
        """
        Execute shortcut command
        
        Args:
            command: Shortcut command text
        """
        self.command_entry.delete(0, tk.END)
        self.command_entry.insert(0, command)
        self.execute_command()
    
    def on_closing(self):
        """
        Handle window closing
        """
        if self.is_running:
            if messagebox.askyesno("Confirm", "Recording is in progress, are you sure you want to exit?"):
                self.stop_recording()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """
    Main function
    """
    import sys
    
    # Ensure proper font display
    if sys.platform == 'win32':
        # Windows system already sets fonts in _setup_fonts
        pass
    
    root = tk.Tk()
    app = MeetingAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()