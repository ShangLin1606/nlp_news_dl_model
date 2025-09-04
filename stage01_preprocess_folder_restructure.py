import os
import shutil

def folder_restructure():
    """
    重新組織特定資料夾內的類別結構
    1. 針對資料夾名稱 `C3-Art`，提取 `Art`
    2. 調整檔案名稱 `C3-Art0002.txt` → `Art0002.txt`
    3. 移動處理後的檔案到新資料夾，並刪除原始 GBK 目錄
    """

    root_dir = 'data'

    # 自動調整 train 和 test 目錄
    for split in ['train', 'test']:
        base_path = os.path.join(root_dir, split)
        for category_folder in os.listdir(base_path):
            old_path = os.path.join(base_path, category_folder)

            # 確認是資料夾
            if os.path.isdir(old_path):
                # 提取類別名稱：C3-Art → Art
                category_name = category_folder.split('-')[-1]  # 取後綴 Art
                delete_start_name = category_folder.split('-')[0] + '-'

                # 建立新的類別資料夾
                new_folder = os.path.join(base_path, category_name)
                os.makedirs(new_folder, exist_ok=True)

                # 處理 utf8 目錄下的檔案
                utf8_path = os.path.join(old_path, 'utf8')
                if os.path.exists(utf8_path):
                    for filename in os.listdir(utf8_path):
                        if filename.startswith(delete_start_name):
                            # 重新命名檔案：C3-Art0002.txt → Art0002.txt
                            new_filename = filename.replace(delete_start_name, '')
                            shutil.move(os.path.join(utf8_path, filename),
                                        os.path.join(new_folder, new_filename))

                    # 刪除 GBK 原始檔案與原資料夾
                    shutil.rmtree(old_path)
                else:
                    print(f"⚠️ 未找到 utf8 目錄：{utf8_path}")

    print("GBK檔案刪除 & 資料夾重構完成！")


def check_folder_restructure():
    # 確認重構後的資料夾結構
    root_dir = 'data'
    for split in ['traditional_train', 'traditional_test']:
        print(f"📂 {split} 資料夾結構：")
        split_path = os.path.join(root_dir, split)
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if os.path.isdir(category_path):
                file_count = len(os.listdir(category_path))
                print(f"  - {category}: {file_count} 篇文章")

if __name__ == "__main__":
    check_folder_restructure()