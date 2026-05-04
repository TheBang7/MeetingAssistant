import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    """
    Meeting Assistant main entry function
    """
    print("=== Meeting Assistant Started ===")
    print("Starting graphical interface...")

    # Start GUI
    from GUI.meeting_assistant_gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
