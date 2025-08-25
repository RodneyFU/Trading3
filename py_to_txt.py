import os

def py_to_txt(root_dir: str = '.', output_file: str = 'all_code.txt'):
    """將所有 PY 檔內容合併到 TXT。
    邏輯：遍歷目錄，讀取 PY 檔，寫入 TXT 並標記檔名。
    """
    with open(output_file, 'w', encoding='utf-8') as out:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(subdir, file)
                    out.write(f"### {path} ###\n")
                    with open(path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                    out.write("\n\n")
    print(f"合併完成：{output_file}")

if __name__ == "__main__":
    py_to_txt()