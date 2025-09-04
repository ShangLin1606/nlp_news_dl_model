import os
import shutil
from pathlib import Path

# 原始和目標目錄
source_base = "data/traditional_train_clean"
target_base = "data/bert_traditional_train_clean_50"

# 確保目標資料夾存在
os.makedirs(target_base, exist_ok=True)

# 遍歷每個類別資料夾
for category_dir in sorted(Path(source_base).iterdir()):
    if not category_dir.is_dir():
        continue  # 忽略非資料夾的項目

    # 找出該類別底下的所有 .pt 檔案，並排序 (可改為 os.listdir() 搭配 filter)
    pt_files = sorted(category_dir.glob("*.txt"))
    # 取前 200 個 .pt 檔案
    selected_files = pt_files[:50]

    # 建立對應類別資料夾
    target_category_dir = Path(target_base) / category_dir.name
    os.makedirs(target_category_dir, exist_ok=True)

    # 開始複製檔案
    for file in selected_files:
        shutil.copy(file, target_category_dir / file.name)

    print(f"類別: {category_dir.name} - 已複製 {len(selected_files)} 個 .txt 檔案")

print("所有類別的前 50 個 .pt 檔案複製完成！")
