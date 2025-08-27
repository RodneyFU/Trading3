import os

def py_to_txt(root_dir: str = '.', output_file: str = 'all_code.txt'):
    """將所有 PY 檔內容合併到 TXT。
    邏輯：遍歷目錄，讀取 PY 檔，寫入 TXT 並標記檔名。
    """
    with open(output_file, 'w', encoding='utf-8') as out:
        jump=""
        exclude = ['venv', 'test']
        for subdir, dirs, files in os.walk(root_dir, topdown=True):
            if len(subdir.lower().split(os.sep))>1 and subdir.lower().split(os.sep)[1] in exclude:
                if jump not in subdir.lower().split(os.sep):
                    jump=subdir.lower().split(os.sep)[1]
                    print(f"跳過資料夾：{jump}")#只出一次
                continue
            for file in files:
                if file.endswith(('.py', '.json')):
                    path = os.path.join(subdir, file)
                    print(f"合併：{path}")
                    out.write(f"### {path} ###\n")
                    with open(path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                    out.write("\n\n")
    print(f"合併完成：{output_file}")

if __name__ == "__main__":
    py_to_txt()