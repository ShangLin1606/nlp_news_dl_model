import os
import glob
from models.llm_model import SummaryModel
from pathlib import Path
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from collections import defaultdict

def process_and_save_summaries(input_base:list, paragraph_max_len:int=2000, no_summary_max_len:int=1024, tokenizer_max_len:int=1024, max_summaries_per_category:int=300):
    """
    針對每個類別進行摘要處理並儲存結果
    1. 若文章長度小於 `no_summary_max_len`，則直接使用原文
    2. 否則，使用 `sliding_window_segment` 進行切割摘要
    3. 最後儲存摘要文本與 tokenized 版本
    """
    
    overlap = paragraph_max_len//4
    model = SummaryModel()
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    category_count = defaultdict(int)

    for input_dir in input_base:
        parent_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(parent_dir, os.path.basename(input_dir) + '_summary')
        os.makedirs(output_dir, exist_ok=True)

        print(parent_dir)
        print(output_dir)

         # 計算已存在的摘要數量
        for existing_file in Path(output_dir).rglob('*.pt'):
            category = existing_file.parent.name
            category_count[category] += 1
        print(category_count)
        

        for file_path in tqdm(glob.glob(f"{input_dir}/**/*.txt", recursive=True)):
            category = Path(file_path).parent.name
            if category_count[category] >= max_summaries_per_category:
                print(f"Skipping category {category}, reached limit {max_summaries_per_category}")
                continue
 
            output_file = Path(output_dir) / Path(file_path).relative_to(input_dir).with_suffix('.pt')
            if output_file.exists():
                print(f"Skipping {output_file}, already exists.")
                continue

            category_count[category] += 1

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()


            
            if len(tokenizer.encode(text=text, truncation=True)) <= no_summary_max_len:
                summary = text
            else:
                segments = sliding_window_segment(text, paragraph_max_len, overlap)
                summary = model.generate_final_summary(segments)
            
            
            # 儲存摘要文本
            relative_path = Path(file_path).relative_to(input_dir)

            category = relative_path.parts[0]


            # 建立新聞類別資料夾
            category_output_dir = Path(output_dir) / category
            category_output_dir.mkdir(parents=True, exist_ok=True)

            output_file = category_output_dir / relative_path.name
            print(output_file)
            
            # output_file = Path(output_dir) / Path(file_path).name
            
            
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            # 將摘要結果轉成 tokens 並儲存
            tokens = tokenizer.encode(summary, max_length=tokenizer_max_len, padding='max_length', truncation=True)
            tokens = torch.tensor(tokens, dtype=torch.long)  # 將 tokens 轉成 tensor
            torch.save(tokens, output_file.with_suffix('.pt'))

def sliding_window_segment(text, max_len, overlap):
    segments = []
    i = 0
    while i < len(text):
        if len(text) - i < max_len:
            segments.append(text[-max_len:])  # 最後一段不足時補齊
            break
        segments.append(text[i:i+max_len])
        i += (max_len - overlap)  # 重疊滑
    
    return segments

if __name__ == "__main__":


    process_and_save_summaries(input_base=['data/traditional_train_clean'], max_summaries_per_category=200)
    process_and_save_summaries(input_base=['data/traditional_test_clean'], max_summaries_per_category=100)