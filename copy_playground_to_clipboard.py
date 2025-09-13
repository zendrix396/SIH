#!/usr/bin/env python3
"""
Script to copy all files from the playground folder to clipboard.
This will recursively go through all files and subfolders in the playground directory
and copy their contents to the system clipboard.
"""

import os
import pyperclip
from pathlib import Path

def get_file_contents(file_path):
    """Read the contents of a file and return as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Error reading {file_path}: {str(e)}\n"

def copy_playground_to_clipboard():
    """Copy all files from playground folder to clipboard."""
    playground_path = Path("playground")
    
    if not playground_path.exists():
        print("Error: 'playground' folder not found in current directory!")
        return
    
    clipboard_content = []
    clipboard_content.append("=" * 80)
    clipboard_content.append("PLAYGROUND RAILWAY SIMULATION - COMPLETE CODEBASE")
    clipboard_content.append("=" * 80)
    clipboard_content.append("")
    
    # Walk through all files in playground directory
    for root, dirs, files in os.walk(playground_path):
        # Sort directories and files for consistent ordering
        dirs.sort()
        files.sort()
        
        for file in files:
            file_path = Path(root) / file
            
            # Skip __pycache__ directories and .pyc files
            if "__pycache__" in str(file_path) or file.endswith(".pyc"):
                continue
                
            # Get relative path from playground folder
            relative_path = file_path.relative_to(playground_path)
            
            clipboard_content.append("")
            clipboard_content.append("=" * 60)
            clipboard_content.append(f"FILE: {relative_path}")
            clipboard_content.append("=" * 60)
            clipboard_content.append("")
            
            # Read and add file contents
            file_contents = get_file_contents(file_path)
            clipboard_content.append(file_contents)
            
            # Add separator if not the last file
            clipboard_content.append("")
            clipboard_content.append("-" * 40)
    
    # Join all content
    final_content = "\n".join(clipboard_content)
    
    # Copy to clipboard
    try:
        pyperclip.copy(final_content)
        print("✅ Successfully copied all playground files to clipboard!")
        print(f"📊 Total characters copied: {len(final_content):,}")
        print(f"📁 Files processed from playground folder")
        print("\nYou can now paste (Ctrl+V) the content anywhere!")
        
    except Exception as e:
        print(f"❌ Error copying to clipboard: {str(e)}")
        print("Make sure you have pyperclip installed: pip install pyperclip")
        
        # Fallback: save to file
        output_file = "playground_codebase.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_content)
            print(f"📄 Content saved to {output_file} instead")
        except Exception as e2:
            print(f"❌ Error saving to file: {str(e2)}")

if __name__ == "__main__":
    print("🚂 Railway Simulation Codebase Copier")
    print("=" * 40)
    copy_playground_to_clipboard()
