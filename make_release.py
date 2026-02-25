import os
import zipfile
import shutil

def write_zip():
    zip_name = "SPAG4D_Portable_Source.zip"
    if os.path.exists(zip_name):
        os.remove(zip_name)
        
    print(f"Creating {zip_name}...")
    
    # Core directories and files to include
    include_dirs = ["spag4d", "static"]
    include_files = [
        "api.py", "install.bat", "run.bat", "pyproject.toml", 
        "requirements.txt", "README.md", ".gitignore"
    ]
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in include_files:
            if os.path.exists(f):
                zf.write(f)
                print(f"Added {f}")
                
        for d in include_dirs:
            if os.path.exists(d):
                for root, dirs, files in os.walk(d):
                    # Skip pycache and large unexpected folders
                    if "__pycache__" in root:
                        continue
                    if "panda_arch\\PanDA" in root:
                        continue
                        
                    for file in files:
                        if file.endswith('.pyc'): continue
                        filepath = os.path.join(root, file)
                        zf.write(filepath)
                print(f"Added dir {d}")
                
    print(f"Success. Zip size: {os.path.getsize(zip_name) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    write_zip()
