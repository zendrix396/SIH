#!/usr/bin/env python3
"""
Simple script to copy all files from the playground folder to a text file.
This will recursively go through all files and subfolders in the playground directory
and save their contents to a single text file.
"""

import os
from pathlib import Path

def get_file_contents(file_path):
    """Read the contents of a file and return as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Error reading {file_path}: {str(e)}\n"

def save_playground_to_file():
    """Save all files from playground folder to a single text file."""
    playground_path = Path("playground")
    
    if not playground_path.exists():
        print("Error: 'playground' folder not found in current directory!")
        return
    
    output_file = "playground_codebase.txt"
    file_contents = []
    
    file_contents.append("=" * 80)
    file_contents.append("PLAYGROUND RAILWAY SIMULATION - COMPLETE CODEBASE")
    file_contents.append("=" * 80)
    file_contents.append("")
    
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
            
            file_contents.append("")
            file_contents.append("=" * 60)
            file_contents.append(f"FILE: {relative_path}")
            file_contents.append("=" * 60)
            file_contents.append("")
            
            # Read and add file contents
            content = get_file_contents(file_path)
            file_contents.append(content)
            
            # Add separator
            file_contents.append("")
            file_contents.append("-" * 40)
    
    # Join all content
    final_content = "\n".join(file_contents)
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print("✅ Successfully saved all playground files!")
        print(f"📄 Output file: {output_file}")
        print(f"📊 Total characters: {len(final_content):,}")
        print(f"📁 Files processed from playground folder")
        print(f"\nYou can now copy the contents of {output_file} to your clipboard!")
        
    except Exception as e:
        print(f"❌ Error saving to file: {str(e)}")

if __name__ == "__main__":
    print("🚂 Railway Simulation Codebase Saver")
    print("=" * 40)
    save_playground_to_file()
