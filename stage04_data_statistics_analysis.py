import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm


def check_data(dir):
    """
    針對指定資料夾統計：
    - 各類別的文章數量
    - 各類別文章的平均字數
    """

    # 計算每類新聞的文章數量與平均字數
    category_data = []

    for category in os.listdir(dir):
        category_path = os.path.join(dir, category)
        if os.path.isdir(category_path):
            lengths = []
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    lengths.append(len(text))
            category_data.append({
                'category': category,
                'article_count': len(lengths),
                'avg_length': int(sum(lengths)/len(lengths)) if lengths else 0
            })
        else:
            print(f'dir:{os.path.isdir(category_path)} error')

    # # 顯示結果
    df = pd.DataFrame(category_data)
    df = df.sort_values(by='article_count', ascending=False)
    df = df.applymap(lambda x: int(x) if isinstance(x, (int, float)) else x)
    print(df)

    # 視覺化類別分佈
    plt.figure(figsize=(10, 5))
    plt.bar(df['Category'], df['Article Count'])
    plt.xticks(rotation=45)
    plt.title('News Category Distribution')
    plt.show()


def check_data_word_count(path): 
    all_texts = []
    lengths = []
    for file in Path(path).glob('*/*.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())

    lengths = [len(text) for text in all_texts]

    data = {
        "統計項目": ["最大字數", "平均字數", "50% 分位數", "60% 分位數", "70% 分位數", "80% 分位數", "90% 分位數"],
        "數值": [
            int(max(lengths)),
            int(sum(lengths)/len(lengths)),
            int(np.percentile(lengths, 50)),
            int(np.percentile(lengths, 60)),
            int(np.percentile(lengths, 70)),
            int(np.percentile(lengths, 80)),
            int(np.percentile(lengths, 90))
        ]
    }

    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    check_list_dirs = ['data/traditional_train_clean', 'data/traditional_test_clean']
    for dir in check_list_dirs:
        print(f'#{dir}####################################')
        check_data(dir)

    # for dir in check_list_dirs:
    #     check_data_word_count(dir)

