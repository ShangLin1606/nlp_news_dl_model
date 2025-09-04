import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bert_score import score 


BERT_MODEL = "bert-base-multilingual-cased" 

MAX_LEN = 512 
STRIDE = int(MAX_LEN * 0.5)  # 每次移動 50% 的長度

def sliding_window_text(text, max_len=MAX_LEN, stride=STRIDE):
    """ 若文本過長，使用移動窗格切割文本 """
    segments = []
    for i in range(0, len(text), stride):
        segments.append(text[i:i + max_len])
        if i + max_len >= len(text):  # 避免超出範圍
            break
    return segments

def calculate_bertscore(original_text, summary_text, model=BERT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"):
    """ 計算 BERTScore 相似度 (Precision, Recall, F1) """
    if len(original_text) > MAX_LEN:
        segments = sliding_window_text(original_text)
        scores = []
        for segment in segments:
            P, R, F1 = score([summary_text], [segment], model_type=model, device=device)
            scores.append({"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()})
        avg_score = {key: np.mean([s[key] for s in scores]) for key in scores[0].keys()}
    else:
        P, R, F1 = score([summary_text], [original_text], model_type=model, device=device)
        avg_score = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}
    return avg_score

def main(input_base, summary_base, output_csv="bert_score_results.csv"):
    """ 逐個類別計算 BERTScore 分數 (支援移動窗格) """
    category_scores = []

    for category in os.listdir(input_base):
        original_files = glob.glob(os.path.join(input_base, category, '*.txt'))
        scores_list = []
        
        for orig_file in tqdm(original_files, desc=f'Processing {category}'):
            summary_file = os.path.join(summary_base, category, Path(orig_file).name)
            if not os.path.exists(summary_file):
                continue  # 如果對應的摘要檔案不存在則跳過

            # 讀取原文與摘要
            with open(orig_file, 'r', encoding='utf-8') as f:
                original_text = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_text = f.read()
            
            # 計算 BERTScore (含移動窗格)
            score_result = calculate_bertscore(original_text, summary_text)
            scores_list.append(score_result)

        # 計算該類別的平均 BERTScore
        avg_category_score = {key: np.mean([s[key] for s in scores_list]) for key in scores_list[0].keys()}
        avg_category_score["Category"] = category  # 加入類別名稱
        category_scores.append(avg_category_score)

    # 存儲結果為 CSV
    df = pd.DataFrame(category_scores)
    df.to_csv(output_csv, index=False)
    print(f"BERTScore 結果已儲存至 {output_csv}")

if __name__ == "__main__":
    # main('data/bert_traditional_train_clean_200', 'data/bert_traditional_train_clean_summary_200')
    main('data/bert_traditional_train_clean', 'data/bert_traditional_train_clean_summary')






# import os
# import glob
# from rouge_score import rouge_scorer
# from tqdm import tqdm
# from pathlib import Path
# import numpy as np

# def calculate_rouge(original_text, summary_text, max_len=8192):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     if len(original_text) > max_len:
#         segments = [original_text[i:i+max_len] for i in range(0, len(original_text), max_len)]
#         scores = []
#         for segment in segments:
#             score = scorer.score(segment, summary_text)
#             scores.append(score)
#         avg_score = {key: np.mean([score[key].fmeasure for score in scores]) for key in scores[0].keys()}
#     else:
#         avg_score = scorer.score(original_text, summary_text)
#     return avg_score

# def main(input_base, summary_base, max_len=8192):
#     category_scores = {}
#     for category in os.listdir(input_base):
#         original_files = glob.glob(os.path.join(input_base, category, '*.txt'))
#         summary_files = glob.glob(os.path.join(summary_base, category, '*.txt'))

#         scores_list = []
#         for orig_file in tqdm(original_files, desc=f'Processing {category}'):
#             summary_file = os.path.join(summary_base, category, Path(orig_file).name)
#             if not os.path.exists(summary_file):
#                 continue

#             with open(orig_file, 'r', encoding='utf-8') as f:
#                 original_text = f.read()

#             with open(summary_file, 'r', encoding='utf-8') as f:
#                 summary_text = f.read()

#             score = calculate_rouge(original_text, summary_text, max_len)
#             scores_list.append(score)

#         avg_category_score = {key: np.mean([score[key] for score in scores_list]) for key in scores_list[0].keys()}
#         category_scores[category] = avg_category_score

#     print("\nFinal ROUGE Scores per Category:")
#     for category, scores in category_scores.items():
#         print(f"{category}: ROUGE-1: {scores['rouge1']:.4f}, ROUGE-2: {scores['rouge2']:.4f}, ROUGE-L: {scores['rougeL']:.4f}")

# if __name__ == "__main__":
#     main('data/test_traditional_test', 'data/test_traditional_test_summary')
