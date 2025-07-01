"""
./data/coin/annotation/COIN.json






"""
import json
path = "./data/msvd/annotation/cap_test.json"
print(f"Path: {path}")
label = input("Enter label: ")

with open(path, "r") as f:
    data = json.load(f)

if label in data:
    print(f"\n🎥 {label}")
    for key, value in data[label].items():
        if key == "annotation":
            print("   annotation:")
            for i, ann in enumerate(value):
                print(f"     - [{i+1}] {ann['label']}, segment: {ann['segment']}")
        else:
            print(f"   {key}: {value}")
else:
    print(f"❌ Could not find '{label}' in {path}")
