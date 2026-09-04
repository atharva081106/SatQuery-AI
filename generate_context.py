import os

def generate_context():
    root_dir = r"d:\sih26167"
    output_file = os.path.join(root_dir, "PROJECT_CONTEXT.md")
    
    ignore_dirs = {".git", "node_modules", ".next", "__pycache__", "venv", "cache", ".huggingface_cache", ".pytest_cache", "public"}
    ignore_exts = {".tif", ".png", ".jpg", ".jpeg", ".ico", ".svg", ".tsbuildinfo", ".json"}
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# SatQuery AI - Project Context\n\n")
        out.write("This file contains the core structure and code of the project to serve as context for LLMs.\n\n")
        
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ignore_exts or file == "package-lock.json" or file == "generate_context.py" or file == "PROJECT_CONTEXT.md":
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                out.write(f"## {rel_path}\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        out.write("```\n")
                        out.write(content)
                        out.write("\n```\n\n")
                except Exception as e:
                    out.write(f"*Could not read file: {e}*\n\n")

if __name__ == "__main__":
    generate_context()
