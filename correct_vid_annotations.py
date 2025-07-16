import os
import json

dataset = "msvd"  # Change if needed

# Paths
annotation_path = f"./data/{dataset}/annotation/qa_train.json"
frames_dir_path = f"./data/{dataset}/frames"
new_annotation_path = annotation_path
changes_log_path = f"./data/{dataset}/annotation/qa_train_changed_frames_log.json"

# Load annotation JSON
print("Loading JSON file...")
with open(annotation_path, "r") as f:
    annotations = json.load(f)
print("Loaded.")

# Track changes
changed_videos = {}

# Update frame_length for each entry
for qid, entry in annotations.items():
    video_name = entry.get("video")
    if not video_name:
        print(f"Warning: Missing 'video' field in entry {qid}")
        continue

    frame_folder = os.path.join(frames_dir_path, video_name)
    print(f"Processing {qid}: video {video_name}")

    if os.path.isdir(frame_folder):
        frame_files = [
            f for f in os.listdir(frame_folder)
            if os.path.isfile(os.path.join(frame_folder, f))
        ]
        num_frames = len(frame_files)

        old_frame_length = entry.get('frame_length', None)
        if old_frame_length != num_frames:
            changed_videos[qid] = {
                "video": video_name,
                "old_frame_length": old_frame_length,
                "new_frame_length": num_frames
            }
            entry['frame_length'] = num_frames
    else:
        print(f"Warning: Frame directory not found for video {video_name} (qid: {qid})")

# Save updated annotation
with open(new_annotation_path, "w") as f:
    json.dump(annotations, f, indent=4)

# Save log of changed frame counts
with open(changes_log_path, "w") as f:
    json.dump(changed_videos, f, indent=4)

print(f"Updated frame_length for all videos.")
print(f"Changed {len(changed_videos)} entries. Log saved to {changes_log_path}")
