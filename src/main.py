import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    """
    会议助手主入口函数
    """
    print("=== 会议助手启动 ===")
    print("正在启动图形界面...")

    # 启动GUI
    from GUI.meeting_assistant_gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
