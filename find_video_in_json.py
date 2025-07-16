import json
file_path="/home/davidg3/MA-LMM/data/msvd/annotation/qa_train.json"
# Load the JSON file
with open(file_path, "r") as f:
    data = json.load(f)

# The video ID you're looking for
target_video = "G6w6kO8UPyg_31_41"

# Find all entries where the video matches
matches = {qid: entry for qid, entry in data.items() if entry.get("video") == target_video}

# Print the results
if matches:
    print(f"Found {len(matches)} match(es) for video ID '{target_video}':\n")
    for qid, entry in matches.items():
        print(f"Question ID: {qid}")
        for key, value in entry.items():
            print(f"  {key}: {value}")
        print()
else:
    print(f"No entry found with video ID '{target_video}'.")