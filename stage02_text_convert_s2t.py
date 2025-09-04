import os
import opencc
from tqdm import tqdm

class TextConverter:
    """
    用於將資料夾中的簡體中文文本轉換為繁體中文，並存入指定目標資料夾。
    """
    def __init__(self, source_folder, target_folder):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.converter = opencc.OpenCC('s2t.json')  # 簡轉繁

    def convert_files(self):
        """
        遍歷來源資料夾，將所有 .txt 文件從簡體轉為繁體，並儲存至目標資料夾。
        """
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

        for category in tqdm(os.listdir(self.source_folder), desc="🔁 轉換中"):
            category_path = os.path.join(self.source_folder, category)
            if os.path.isdir(category_path):
                target_category_path = os.path.join(self.target_folder, category)
                os.makedirs(target_category_path, exist_ok=True)

                for file_name in os.listdir(category_path):
                    source_file = os.path.join(category_path, file_name)
                    target_file = os.path.join(target_category_path, file_name)

                    with open(source_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    text_traditional = self.converter.convert(text)

                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(text_traditional)

        print("所有檔案已轉換為繁體！")


if __name__ == "__main__":
    # source_dir = "data/train"
    # target_dir = "data/traditional_train"
    source_dir = "data/test"
    target_dir = "data/traditional_test"
    converter = TextConverter(source_dir, target_dir)
    converter.convert_files()


    