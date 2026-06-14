import os, sys, shutil, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

candidates = glob.glob("training/runs/fire_smoke_fast*/weights/best.pt")
if not candidates:
    print("ERROR: No trained weights found.")
    print("       Run first: python training\\train_fast.py")
    sys.exit(1)

src = candidates[0]
dst = "PyroWatch/detectors/yolo_fire/fire_smoke_best.pt"
os.makedirs("PyroWatch/detectors/yolo_fire", exist_ok=True)
shutil.copy2(src, dst)
size = round(os.path.getsize(dst)/1024/1024, 1)
print(f"Deployed: {dst}  ({size} MB)")
print()
print("Update run.py -- change the import line to:")
print("  from ifsd.detectors.yolo_fire.detector import YOLOFireDetector as FireDetector")
print("  fire_det = FireDetector()")
print()
print("Then run: python run.py")



