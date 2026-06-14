import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

api_key = None
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("ROBOFLOW_API_KEY"):
                api_key = line.split("=", 1)[1].strip()

if not api_key:
    print("ERROR: No API key found in .env")
    sys.exit(1)

print(f"API key loaded.")

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)

print("Trying dataset download -- Middle East Tech University fire+smoke dataset...")
try:
    project = rf.workspace("middle-east-tech-university").project("fire-and-smoke-detection-hiwia")
    versions = project.versions()
    print(f"Available versions: {[v.version for v in versions]}")
    dataset = project.version(versions[0].version).download(
        "yolov8",
        location="training/dataset",
        overwrite=True
    )
    print("SUCCESS")
except Exception as e:
    print(f"Failed: {e}")
    print()
    print("Trying backup dataset...")
    try:
        project = rf.workspace("fands").project("fire-and-smoke-hjs0i")
        dataset = project.version(4).download(
            "yolov8",
            location="training/dataset",
            overwrite=True
        )
        print("SUCCESS with backup dataset")
    except Exception as e2:
        print(f"Backup also failed: {e2}")



