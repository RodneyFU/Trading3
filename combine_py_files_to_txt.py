import os
from pathlib import Path
import logging
import argparse
from datetime import datetime
import glob

# 設置日誌，保留英文訊息以便技術除錯
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_files_to_txt(file_patterns: list, output_file: str = "combined_files.txt"):
    """
    將指定的 .py 和 config/*.json 檔案內容合併到一個或多個 .txt 檔案。
    功能：
    - 根據輸入的檔案模式（支援 *.py、config/*.json 或具體檔案名稱），讀取符合條件的檔案。
    - 排除 combine_py_files_to_txt.py 自身。
    - 當內容超過 50,000 字符時，自動分割為多個輸出檔案，確保單一檔案內容不被分割。
    - 輸出簡化的中文控制台訊息，保留英文日誌供除錯。
    參數：
    - file_patterns: 檔案模式或路徑清單。
    - output_file: 輸出 .txt 檔案名稱，預設為 'combined_files.txt'。
    """
    try:
        # 獲取當前資料夾和 config 資料夾路徑
        current_dir = Path.cwd()
        config_dir = current_dir / "config"
        logging.info(f"Starting to process files, current directory: {current_dir}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始處理檔案")

        # 收集所有要處理的檔案
        file_paths = []
        for pattern in file_patterns:
            if pattern == "*.py":
                # 收集當前資料夾的所有 .py 檔案，排除自身
                py_files = [f for f in current_dir.glob("*.py") if f.name != "combine_py_files_to_txt.py"]
                file_paths.extend(py_files)
                logging.info(f"Collected {len(py_files)} .py files")
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 收集到 {len(py_files)} 個 .py 檔案")
            elif pattern == "config/*.json":
                # 收集 config 資料夾的所有 .json 檔案
                if config_dir.exists():
                    json_files = list(config_dir.glob("*.json"))
                    file_paths.extend(json_files)
                    logging.info(f"Collected {len(json_files)} config/*.json files")
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 收集到 {len(json_files)} 個 config/*.json 檔案")
                else:
                    logging.warning("Config directory does not exist, cannot collect config/*.json")
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} config 資料夾不存在")
            else:
                # 處理具體檔案名稱
                file_path = Path(pattern)
                if file_path.exists() and file_path.suffix in ['.py', '.json']:
                    file_paths.append(file_path)
                    logging.info(f"Added specific file: {file_path.name}")
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 添加檔案：{file_path.name}")
                else:
                    logging.warning(f"File {pattern} does not exist or is of unsupported type (only .py and .json allowed)")
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 檔案 {pattern} 不存在或類型不受支援")

        # 移除重複檔案並排序
        file_paths = sorted(list(set(file_paths)))
        if not file_paths:
            logging.warning("No valid files to process (only *.py and config/*.json allowed, excluding combine_py_files_to_txt.py)")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無有效檔案可處理")
            return

        # 初始化輸出檔案管理
        output_index = 1
        current_char_count = 0
        max_chars = 50000  # 最大字符數限制
        base_output_file = Path(output_file).stem
        output_ext = Path(output_file).suffix
        txt_file = open(f"{base_output_file}{output_index}{output_ext}", 'w', encoding='utf-8')
        txt_file.write(f"檔案合併開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        logging.info(f"Opened output file: {base_output_file}{output_index}{output_ext}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始寫入檔案：{base_output_file}{output_index}{output_ext}")

        for file_path in file_paths:
            try:
                # 驗證檔案類型
                if file_path.suffix not in ['.py', '.json']:
                    error_msg = f"File {file_path.name} is of unsupported type (only .py and .json allowed)"
                    logging.warning(error_msg)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 檔案 {file_path.name} 類型不受支援")
                    txt_file.write(f"\n===== File: {file_path.name} =====\n")
                    txt_file.write(f"錯誤：檔案類型不受支援\n")
                    current_char_count += len(f"\n===== File: {file_path.name} =====\n錯誤：檔案類型不受支援\n")
                    continue

                # 驗證 JSON 檔案是否在 config 資料夾
                if file_path.suffix == '.json' and file_path.parent != config_dir:
                    error_msg = f"JSON file {file_path.name} is not in config directory"
                    logging.warning(error_msg)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} JSON 檔案 {file_path.name} 不在 config 資料夾")
                    txt_file.write(f"\n===== File: {file_path.name} =====\n")
                    txt_file.write(f"錯誤：JSON 檔案必須位於 config 資料夾\n")
                    current_char_count += len(f"\n===== File: {file_path.name} =====\n錯誤：JSON 檔案必須位於 config 資料夾\n")
                    continue

                # 讀取檔案內容
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # 計算即將寫入的內容長度
                content_to_write = f"\n===== File: {file_path.name} =====\n{content}\n"
                content_length = len(content_to_write)

                # 檢查是否需要開新檔案
                if current_char_count + content_length > max_chars:
                    txt_file.close()
                    output_index += 1
                    txt_file = open(f"{base_output_file}{output_index}{output_ext}", 'w', encoding='utf-8')
                    txt_file.write(f"檔案合併開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    logging.info(f"Opened new output file: {base_output_file}{output_index}{output_ext}")
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 內容超出限制，開始寫入新檔案：{base_output_file}{output_index}{output_ext}")
                    current_char_count = 0

                # 寫入檔案名稱分隔線和內容
                txt_file.write(content_to_write)
                current_char_count += content_length
                logging.info(f"Processed file: {file_path.name}")
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已處理檔案：{file_path.name}")

            except Exception as e:
                error_msg = f"Failed to read file {file_path.name}: {str(e)}"
                logging.error(error_msg)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無法讀取檔案 {file_path.name}")
                error_content = f"\n===== File: {file_path.name} =====\n錯誤：無法讀取檔案 - {str(e)}\n"
                txt_file.write(error_content)
                current_char_count += len(error_content)

        # 關閉最後的輸出檔案
        txt_file.close()
        logging.info(f"All specified files have been merged to output files starting with {base_output_file}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 所有指定檔案已合併完成")

    except Exception as e:
        error_msg = f"Failed to merge files: {str(e)}"
        logging.error(error_msg)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 合併檔案失敗")
        if 'txt_file' in locals():
            txt_file.close()

if __name__ == "__main__":
    # 設置命令列參數解析
    parser = argparse.ArgumentParser(description="將指定的 .py 和 config/*.json 檔案合併到一個或多個 .txt 檔案")
    parser.add_argument('files', nargs='*', help="要合併的檔案模式或路徑（支援 *.py, config/*.json 或具體檔案名稱），留空則處理所有符合條件的檔案")
    parser.add_argument('--output', default="combined_files.txt", help="輸出 .txt 檔案名稱，預設為 'combined_files.txt'")
    args = parser.parse_args()

    # 如果未提供檔案，預設處理 *.py 和 config/*.json
    file_patterns = args.files if args.files else ["*.py", "config/*.json"]

    # 執行合併
    combine_files_to_txt(file_patterns, args.output)