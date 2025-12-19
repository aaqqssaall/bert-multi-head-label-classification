import json
import csv
import glob

rows = []

for filename in glob.glob("*.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            text = item.get("text")
            if text:
                cleaned = text.replace("\n", " ").replace("\r", " ")
                rows.append([cleaned])

with open("newdata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
