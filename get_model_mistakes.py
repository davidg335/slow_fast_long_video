import json
import os

# Paths
qa_test_path = "/home/davidg3/MA-LMM/data/msvd/annotation/qa_test.json"
model_guess_path = "/home/davidg3/MA-LMM/lavis/output/msvd_qa/blip2_vicuna_instruct_vicuna7b/test/b8_e5_lr0.0001_wd0.05_q32_f20_fb10_freezevit/result/beam5/test_epoch2_rank0.json"
output_dir = "/home/davidg3/MA-LMM/logs/model_mistakes/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "mistakes_test_epoch2_rank0.json")

# Load ground truth
with open(qa_test_path, 'r') as f:
    gt_data = json.load(f)

# Load model predictions
with open(model_guess_path, 'r') as f:
    pred_data = json.load(f)

# Collect mistakes
mistakes = {}

for entry in pred_data:
    qid = entry['question_id']
    model_answers = [ans.strip().lower() for ans in entry.get('answer', [])]

    # Skip if question_id not in ground truth
    if qid not in gt_data:
        continue

    gt_answer = gt_data[qid]['answer'].strip().lower()

    if gt_answer not in model_answers:
        mistakes[qid] = {
            'question': gt_data[qid]['question'],
            'ground_truth_answer': gt_answer,
            'model_guesses': model_answers
        }

# Save mistakes
with open(output_path, 'w') as f:
    json.dump(mistakes, f, indent=2)

print(f"Saved {len(mistakes)} mistakes to {output_path}")
