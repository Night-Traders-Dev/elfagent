import sqlite3
import json
import os

DB_PATH = "../data/dbs/shared_engram.sqlite"
OUTPUT_FILE = "../data/dpo_training_data.jsonl"

def export_dpo_dataset():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found. Has ElfAgent collected any DPO data yet?")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT prompt, chosen, rejected FROM dpo_dataset")
        rows = cursor.fetchall()

        if not rows:
            print("No DPO records found in the database yet.")
            return

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for row in rows:
                prompt, chosen, rejected = row
                # HuggingFace TRL DPO format
                record = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                }
                f.write(json.dumps(record) + '\n')

        print(f"âœ… Successfully exported {len(rows)} preference pairs to {OUTPUT_FILE}")
        print("Data is now ready for HuggingFace TRL (trl.DPOTrainer) fine-tuning!")

    except Exception as e:
        print(f"Export failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    export_dpo_dataset()
