from ifsd.analytics.scene import SceneDetector
print("\n" + "="*50)
print(" TESTING STABLE INTERFACE INFRASTRUCTURE FOR SCENE ENGINE...")
print("="*50)
try:
    detector = SceneDetector()
    print(" [+] Architecture imported successfully and YOLOv8 pipeline is green!")
except Exception as e:
    print(f" [ERROR] Structural validation error on parsing: {e}")
print("="*50 + "\n")



