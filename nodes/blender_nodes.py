import subprocess
import os
from pathlib import Path
import folder_paths as folder_paths

file_directory = os.path.dirname(os.path.abspath(__file__))
scripts_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)),'scripts')

class VisualBrunoToolsFBXRenameToSMPL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fbx_file":("STRING",),
                "fbx_output_path":("STRING",),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("fbx_output_file", )
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/Blender"
    OUTPUT_NODE = True

    def process(self, fbx_file, fbx_output_path, ):        
        blender_path = os.environ["BLENDER_EXE"]
        
        env = os.environ.copy()
        env["INPUT_MESH"] = fbx_file
        
        fbx_output_path = self.prepare_full_path(fbx_output_path, folder_paths.get_output_directory())
        
        env["OUTPUT_MESH"] = fbx_output_path      
        
        script_path = os.path.join(scripts_directory,'RenameToSMPL.py')
        
        print('Running Blender ...')
        command = [
            blender_path,
            "-b",               # Run in background
            "--python", script_path
        ]

        subprocess.run(command, env=env)        
        
        return (fbx_output_path,)

    def prepare_full_path(self, user_input, default_dir):
        clean_input = user_input.strip().strip('"').strip("'")
        
        # 1. Determine the full path
        if not os.path.dirname(clean_input):
            # User only gave "filename.txt"
            full_path = os.path.join(default_dir, clean_input)
        else:
            # User gave "folder/filename.txt" or "C:/folder/filename.txt"
            full_path = os.path.abspath(clean_input)

        # 2. Extract the directory portion of that path
        target_dir = os.path.dirname(full_path)

        # 3. Create the directory if it doesn't exist
        if target_dir and not os.path.exists(target_dir):
            try:
                # exist_ok=True prevents errors if another process creates it simultaneously
                os.makedirs(target_dir, exist_ok=True)
                print(f"Created directory: {target_dir}")
            except Exception as e:
                print(f"Error creating directory {target_dir}: {e}")
                
        return full_path       
