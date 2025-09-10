import pyperclip
import os

def copy_code_to_clipboard():
    """
    Reads all Python files from the current directory and the 'src' directory,
    formats them with file path headers, and copies the combined content
    to the system clipboard.
    """
    clipboard_content = ""
    
    # Define the files and directories to include
    files_to_copy = ['agent.py']
    dirs_to_copy = ['src']

    # Process standalone files
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            clipboard_content += f"--- FILE: {file_path} ---\n\n"
            with open(file_path, 'r') as f:
                clipboard_content += f.read()
            clipboard_content += "\n\n"
            
    # Process directories
    for directory in dirs_to_copy:
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.py'):
                    file_path = os.path.join(directory, filename)
                    clipboard_content += f"--- FILE: {file_path} ---\n\n"
                    with open(file_path, 'r') as f:
                        clipboard_content += f.read()
                    clipboard_content += "\n\n"

    try:
        pyperclip.copy(clipboard_content)
        print("="*50)
        print("✅ Code copied to clipboard successfully!")
        print("="*50)
    except pyperclip.PyperclipException:
        print("="*50)
        print("❌ Error: Could not copy to clipboard.")
        print("Please make sure you have a clipboard utility installed.")
        print("You can install one with:")
        print("  - on Linux: sudo apt-get install xclip or sudo apt-get install xsel")
        print("  - on Windows/Mac, this should not be an issue.")
        print("="*50)


if __name__ == "__main__":
    copy_code_to_clipboard()
