import argparse
import sys
import os
import shutil
from pathlib import Path

def get_templates_dir():
    # .trae/skills/cpp_toolkit/scripts/templates
    return Path(__file__).resolve().parent / "templates"

def init_project(args):
    target_dir = Path(args.dir).resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[Error] Target directory {target_dir} is not empty.")
        sys.exit(1)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    src_dir = target_dir / "src"
    inc_dir = target_dir / "include"
    src_dir.mkdir(exist_ok=True)
    inc_dir.mkdir(exist_ok=True)

    templates_dir = get_templates_dir()

    if args.template == "trt_yolo":
        # Scaffold a standalone YOLO TRT project
        print(f"[Info] Scaffolding Standalone YOLO TRT Project at {target_dir}")
        # In a real implementation, we would copy from templates_dir / "4_deployment_tensorrt" / "models"
        # For now, generate the CMakeLists.txt
        cmake_content = f"""cmake_minimum_required(VERSION 3.10)
project({target_dir.name} CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

# TensorRT & OpenCV (Modify as needed)
set(TENSORRT_DIR "/usr/local/TensorRT")
include_directories(${{TENSORRT_DIR}}/include)
link_directories(${{TENSORRT_DIR}}/lib)
find_package(OpenCV REQUIRED)
include_directories(${{OpenCV_INCLUDE_DIRS}})
include_directories(${{CMAKE_CURRENT_SOURCE_DIR}}/include)

add_executable(app src/main.cpp)
target_link_libraries(app nvinfer nvparsers cudart ${{OpenCV_LIBS}})
"""
        with open(target_dir / "CMakeLists.txt", "w") as f:
            f.write(cmake_content)
        
        with open(src_dir / "main.cpp", "w") as f:
            f.write("// Auto-generated standalone entry point\n#include <iostream>\n\nint main() {\n    std::cout << \"TRT App Initialized\" << std::endl;\n    return 0;\n}\n")
        
        print("[Success] Project scaffolded. Remember to copy the specific TRT CUDA kernels into src/.")
        
    elif args.template == "basic":
        print(f"[Info] Scaffolding Basic C++ Project at {target_dir}")
        with open(target_dir / "CMakeLists.txt", "w") as f:
            f.write(f"cmake_minimum_required(VERSION 3.10)\nproject({target_dir.name} CXX)\nadd_executable(app src/main.cpp)\n")
        with open(src_dir / "main.cpp", "w") as f:
            f.write("#include <iostream>\n\nint main() {\n    std::cout << \"Hello World\" << std::endl;\n    return 0;\n}\n")
        print("[Success] Project scaffolded.")
    else:
        print(f"[Error] Unknown template: {args.template}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="C++ Toolkit CLI Router (P10 Architecture Enforcer)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser("init", help="Initialize a completely decoupled standalone C++ project")
    init_parser.add_argument("--template", choices=["basic", "trt_yolo", "cuda_gemm"], required=True, help="Project template to scaffold")
    init_parser.add_argument("--dir", required=True, help="Target directory for the standalone project")

    args = parser.parse_args()

    if args.command == "init":
        init_project(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
