"""
updates the json file to have the correct number of frames for each video 

"""

import os
import json

# Paths
annotation_path = "./data/breakfast/annotation/val.json"
frames_dir_path = "./data/breakfast/frames"
new_annotation_path = "./data_old/breakfast/annotation/val.json"

# Load annotation JSON
print("Loading JSON file...")
with open(annotation_path, "r") as f:
    annotations = json.load(f)
print("Loaded")

# Update frame_length for each video
for video_name in annotations:
    frame_folder = os.path.join(frames_dir_path, video_name)
    print(f"video {video_name}")
    if os.path.isdir(frame_folder):
        num_frames = len([f for f in os.listdir(frame_folder) if os.path.isfile(os.path.join(frame_folder, f))])
        annotations[video_name]['frame_length'] = num_frames
    else:
        print(f"Warning: Frame directory not found for {video_name}")

# Save the updated JSON
with open(new_annotation_path, "w") as f:
    json.dump(annotations, f, indent=4)

print("Updated frame_length for all videos.")
