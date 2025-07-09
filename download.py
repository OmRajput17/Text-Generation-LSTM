import requests

DATA_PATH = "data/pride_and_prejudice.txt"
URL = "https://www.gutenberg.org/files/1342/1342-0.txt"

raw_text = requests.get(URL).text
print(len(raw_text))
with open(DATA_PATH, "w", encoding="utf-8") as f:
    f.write(raw_text)
