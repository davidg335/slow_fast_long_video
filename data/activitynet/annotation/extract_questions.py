import json
import csv

# Load the JSON file
with open('qa_val.json', 'r') as f:
    data = json.load(f)

# Output CSV file name
output_csv = 'filtered_questions_TIEzvhv6xaI.csv'

# Filter entries by video ID and extract required fields
filtered_data = []
for key, entry in data.items():
    if entry.get('video') == 'TIEzvhv6xaI':
        filtered_data.append({
            'question': entry.get('question', ''),
            'answer': entry.get('answer', ''),
            'answer_type': entry.get('answer_type', '')
        })

# Save to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['question', 'answer', 'answer_type'])
    writer.writeheader()
    writer.writerows(filtered_data)

print(f"Saved {len(filtered_data)} entries to {output_csv}")
