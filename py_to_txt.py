import os

def py_to_txt(root_dir: str = '.', output_file: str = 'all_code.txt'):
    """將所有 PY 檔內容合併到 TXT。
    邏輯：遍歷目錄，讀取 PY 檔，寫入 TXT 並標記檔名。
    """

    with open(output_file, 'w', encoding='utf-8') as out:
        exclude = ['venv', 'test']
        for subdir, dirs, files in os.walk(root_dir, topdown=True):
            for jump in exclude:
                if subdir==root_dir and jump in dirs:
                    dirs.remove(jump)
                    print(f"跳過根目錄的資料夾：{jump}")
            for file in files:
                if file.endswith(('.py', '.json')) and not file.endswith(('py.py')) and not file.startswith(('grafana_dashboard','api_key','data_fetcher','model')):
                    path = os.path.join(subdir, file)
                    print(f"合併：{path}")
                    out.write(f"---### {path} ###---\n")
                    with open(path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                    out.write("\n\n")
    print(f"合併完成：{output_file}")

if __name__ == "__main__":
    py_to_txt()