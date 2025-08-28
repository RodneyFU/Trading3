import os
import re

def txt_to_py(input_file: str = 'all_code.txt', root_dir: str = '.'):
    """將合併的 TXT 檔還原為原始的 PY 和 JSON 檔案。
    邏輯：讀取 TXT 檔，根據 ### 標記分割內容，還原到原始路徑。
    """
    if not os.path.exists(input_file):
        print(f"錯誤：找不到輸入檔案 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正規表達式分割檔案內容，匹配 ### 路徑 ### 標記
    sections = re.split(r'###\s+(.+?)\s+###\n', content)[1:]  # 跳過開頭空內容

    # 每兩個元素為一組（路徑, 內容）
    for i in range(0, len(sections), 2):
        file_path = sections[i].strip()
        file_content = sections[i + 1].strip()

        # 確保目標目錄存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"建立目錄：{directory}")

        # 寫入檔案
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            print(f"還原檔案：{file_path}")
        except Exception as e:
            print(f"寫入檔案 {file_path} 失敗：{e}")

    print(f"還原完成：所有檔案已從 {input_file} 還原到原始路徑")

if __name__ == "__main__":
    txt_to_py()
