#!/usr/bin/env python3
"""
Validation script for Educational AI & Humanoid Robotics Platform
This script verifies that all components of the platform are properly set up.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_environment():
    """Check if Python environment meets requirements"""
    print("Checking Python environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        return False
    else:
        print(f"[SUCCESS] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

    # Check if in virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

    if in_venv:
        print("[SUCCESS] Running in virtual environment")
    else:
        print("[WARNING] Not running in virtual environment (recommended but not required)")

    return True

def check_directories():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "docs",
        "backend",
        "backend/api",
        "backend/agents", 
        "backend/rag",
        "backend/embeddings",
        "backend/tests",
        "frontend",
        "docs/ros",
        "docs/vlm", 
        "docs/simulation",
        "docs/humanoid-robotics",
        "docs/exercises",
        "docs/projects"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"[SUCCESS] {dir_path}/")
        else:
            print(f"[ERROR] {dir_path}/ - MISSING")
            all_exist = False

    return all_exist

def check_documentation_files():
    """Check if required documentation files exist"""
    print("\nChecking documentation files...")

    required_docs = [
        "docs/intro.md",
        "docs/ros/fundamentals.md",
        "docs/ros/nodes-topics-services.md",
        "docs/ros/launch-files.md",
        "docs/ros/packages-workspaces.md",
        "docs/vlm/introduction.md",
        "docs/vlm/vla-architectures.md",
        "docs/vlm/embedding-techniques.md",
        "docs/vlm/planning-with-vlm.md",
        "docs/simulation/gazebo.md",
        "docs/simulation/isaac-sim.md",
        "docs/simulation/unity-robotics.md",
        "docs/humanoid-robotics/introduction.md",
        "docs/humanoid-robotics/kinematics.md",
        "docs/humanoid-robotics/control-systems.md",
        "docs/humanoid-robotics/walking-algorithms.md",
        "docs/humanoid-robotics/human-robot-interaction.md",
        "docs/exercises/chapter1.md",
        "docs/exercises/chapter2.md",
        "docs/exercises/chapter3.md",
        "docs/projects/project1.md",
        "docs/projects/project2.md",
        "docs/projects/project3.md"
    ]

    all_exist = True
    for doc_path in required_docs:
        if Path(doc_path).exists():
            print(f"[SUCCESS] {doc_path}")
        else:
            print(f"[ERROR] {doc_path} - MISSING")
            all_exist = False

    return all_exist

def check_backend_components():
    """Check backend components"""
    print("\nChecking backend components...")

    required_backend_files = [
        "backend/__init__.py",
        "backend/api/main.py",
        "backend/api/config.py",
        "backend/agents/coordinator.py",
        "backend/agents/ros2_subagent.py",
        "backend/agents/vlm_agent.py",
        "backend/agents/simulation_agent.py",
        "backend/agents/writer_agent.py",
        "backend/rag/knowledge_rag.py",
        "backend/embeddings/processor.py",
        "backend/requirements.txt",
        "backend/start_server.py"
    ]

    all_exist = True
    for file_path in required_backend_files:
        if Path(file_path).exists():
            print(f"[SUCCESS] {file_path}")
        else:
            print(f"[ERROR] {file_path} - MISSING")
            all_exist = False

    return all_exist

def check_frontend_components():
    """Check frontend components"""
    print("\nChecking frontend components...")

    required_frontend_files = [
        "package.json",
        "docusaurus.config.js",
        "sidebars.js"
    ]

    all_exist = True
    for file_path in required_frontend_files:
        if Path(file_path).exists():
            print(f"[SUCCESS] {file_path}")
        else:
            print(f"[ERROR] {file_path} - MISSING")
            all_exist = False

    return all_exist

def check_dependencies():
    """Check if dependencies are available"""
    print("\nChecking dependencies...")
    
    # Check if required Python packages can be imported
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("qdrant_client", "Qdrant client"),
        ("langchain", "LangChain")
    ]
    
    all_importable = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"[SUCCESS] {name} available")
        except ImportError:
            print(f"[ERROR] {name} - NOT INSTALLED")
            all_importable = False

    return all_importable

def check_docker_files():
    """Check Docker-related files"""
    print("\nChecking Docker configuration...")
    
    docker_files = [
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    all_exist = True
    for file_path in docker_files:
        if Path(file_path).exists():
            print(f"[SUCCESS] {file_path}")
        else:
            print(f"[WARNING] {file_path} - NOT FOUND (optional)")

    return True  # Docker files are optional

def check_config_files():
    """Check configuration files"""
    print("\nChecking configuration files...")
    
    config_files = [
        "docusaurus.config.js",
        "sidebars.js",
        "README.md",
        "environment.yml"
    ]
    
    all_exist = True
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"[SUCCESS] {file_path}")
        else:
            print(f"[ERROR] {file_path} - MISSING")
            all_exist = False

    return all_exist

def validate_sidebar_links():
    """
    Validate that sidebar references actual files
    Properly handles Docusaurus category structure
    """
    print("\nValidating sidebar links...")

    # Read sidebars.js
    try:
        with open("sidebars.js", "r") as f:
            sidebar_content = f.read()
    except FileNotFoundError:
        print("[ERROR] sidebars.js - MISSING")
        return False

    # Properly extract only actual document references, not category labels
    import re

    # Find content between items: [ ... ] to get document refs
    items_pattern = r"items:\s*\[([^\]]+)\]"
    matches = re.findall(items_pattern, sidebar_content)

    doc_refs = []
    for match in matches:
        # Find all quoted strings inside the items array (these are actual doc refs)
        item_refs = re.findall(r"'([^']*)'", match)
        doc_refs.extend(item_refs)

    # Also find the intro reference outside of items arrays
    intro_match = re.search(r"'intro'", sidebar_content)
    if intro_match:
        doc_refs.append('intro')

    # Add only actual document references, not category names
    for ref in doc_refs:
        if ref not in ['category', 'type'] and '/' not in ref:  # Exclude category declarations
            continue  # For valid document refs, we keep them

    all_valid = True
    for ref in doc_refs:
        # Convert sidebar reference to file path
        if not ref.endswith('.md'):
            ref += '.md'

        # Check if file exists
        if Path(f"docs/{ref}").exists():
            print(f"[SUCCESS] docs/{ref}")
        else:
            print(f"[ERROR] docs/{ref} - MISSING")
            all_valid = False

    return all_valid

def main():
    """Main validation function"""
    print("="*60)
    print("Educational AI & Humanoid Robotics Platform - Validation")
    print("="*60)
    
    all_checks_passed = True
    
    # Run all checks
    checks = [
        ("Python Environment", check_python_environment),
        ("Directory Structure", check_directories),
        ("Documentation Files", check_documentation_files),
        ("Backend Components", check_backend_components),
        ("Frontend Components", check_frontend_components),
        ("Dependencies", check_dependencies),
        ("Docker Configuration", check_docker_files),
        ("Configuration Files", check_config_files),
        ("Sidebar Links", validate_sidebar_links)
    ]
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_checks_passed = False
        except Exception as e:
            print(f"[ERROR] Error during {check_name} check: {e}")
            all_checks_passed = False

    print("\n" + "="*60)
    if all_checks_passed:
        print("[SUCCESS] ALL CHECKS PASSED! Platform is ready for use.")
        print("\nThe Educational AI & Humanoid Robotics Platform is properly configured.")
        print("You can now:")
        print("  1. Start the backend: cd backend && python start_server.py")
        print("  2. Start the documentation: npm start")
        print("  3. Access the platform and begin learning!")
    else:
        print("[ERROR] SOME CHECKS FAILED. Please review the above issues.")
        print("\nFor assistance, refer to the README.md for setup instructions.")
    print("="*60)

    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)