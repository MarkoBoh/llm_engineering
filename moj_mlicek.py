import subprocess
import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

import platform
import psutil
import torch

def get_system_info():
    # CPU Information
    cpu_info = {
        "Processor": platform.processor(),
        "Cores": psutil.cpu_count(logical=False),
        "Logical Cores": psutil.cpu_count(logical=True),
        "Max Frequency (GHz)": psutil.cpu_freq().max / 1000
    }

    # RAM Information
    ram_info = {
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2)
    }

    # GPU Information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "GPU Name": torch.cuda.get_device_name(0),
            "CUDA Capability": torch.cuda.get_device_capability(0),
            "GPU RAM (GB)": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        }
    else:
        gpu_info = {"GPU": "No CUDA-compatible GPU found"}

    return {"CPU": cpu_info, "RAM": ram_info, "GPU": gpu_info}

def evaluate_llm_suitability(system_info):
    evaluation = []

    # CPU Evaluation
    if system_info["CPU"]["Cores"] >= 4 and system_info["CPU"]["Max Frequency (GHz)"] >= 2.5:
        evaluation.append("CPU: Suitable for small to medium LLMs.")
    else:
        evaluation.append("CPU: May struggle with large LLMs; consider an upgrade.")

    # RAM Evaluation
    if system_info["RAM"]["Total RAM (GB)"] >= 16:
        evaluation.append("RAM: Sufficient for small to medium LLMs.")
    elif system_info["RAM"]["Total RAM (GB)"] >= 8:
        evaluation.append("RAM: Suitable for small LLMs, but might need optimization for larger models.")
    else:
        evaluation.append("RAM: Insufficient for most LLMs; consider an upgrade.")

    # GPU Evaluation
    if "GPU RAM (GB)" in system_info["GPU"]:
        if system_info["GPU"]["GPU RAM (GB)"] >= 12:
            evaluation.append("GPU: Suitable for medium to large LLMs.")
        elif system_info["GPU"]["GPU RAM (GB)"] >= 8:
            evaluation.append("GPU: Suitable for small to medium LLMs.")
        else:
            evaluation.append("GPU: Insufficient for most LLMs; consider an upgrade.")
    else:
        evaluation.append("GPU: Not suitable for LLMs requiring GPU acceleration.")

    return evaluation

if __name__ == "__main__":
    system_info = get_system_info()
    print("System Information:")
    for component, details in system_info.items():
        print(f"{component}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

    print("\nLLM Suitability Evaluation:")
    suitability = evaluate_llm_suitability(system_info)
    for recommendation in suitability:
        print(f"- {recommendation}")
