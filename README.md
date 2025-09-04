# 專案名稱：新聞分類與摘要系統

## 1. 專案概述
本專案旨在開發一個基於深度學習的新聞分類與摘要系統，利用 **BAAI/bge-m3** 模型進行新聞文本嵌入，並使用 **BiLSTM** 進行文本分類。本系統包含從資料預處理、統計分析、文本摘要生成、摘要評估、資料集製作、模型訓練等完整流程。

## 2. 資料來源
- **數據集名稱**：Categorised News Dataset from Fudan University
- **來源**：Kaggle
- **內容**：該數據集包含來自不同領域的新聞報導，並已依照類別標記，適用於文本分類與摘要生成的研究。

## 3. 專案處理流程
本專案依照 **數據處理 → 訓練 → 評估** 的方式，進行新聞文本的自動摘要與分類。

### (1) 資料預處理與結構化
1. **`stage01_preprocess_folder_restructure.py`**：
   - 重新整理原始新聞數據的資料夾結構，確保數據組織良好。
   
2. **`stage02_text_convert_s2t.py`**：
   - 進行簡繁體轉換，確保所有文本統一為繁體中文。
   
3. **`stage03_text_cleaning.py`**：
   - 清理文本，如去除多餘的空白與雜訊，確保模型輸入的乾淨程度。
   
### (2) 資料統計分析
4. **`stage04_data_statistics_analysis.py`**：
   - 計算新聞文本長度分佈，並提供摘要策略建議（如最佳 `max_len` 設置）。
   
### (3) 生成摘要與摘要評估
5. **`stage05_generate_summary.py`**：
   - 利用 **LLM** 進行新聞摘要生成，並存儲摘要文本與對應 tokens。
   
6. **`stage06_summary_evaluation_bertscore.py`**：
   - 使用**BERTScore** 評估摘要品質。
   
### (4) 資料集準備與模型訓練
7. **`stage07_dataset_preparation.py`**：
   - 準備 **BiLSTM** 訓練所需的數據集，並存為 `.pt` 格式。
   
8. **`stage08_model_training.py`**：
   - 使用 **BAAI/bge-m3 + BiLSTM** 進行新聞分類模型訓練，並評估模型效能（Accuracy、Precision、Recall、F1-score）。

## 4. 模型技術細節
- **嵌入模型**：BAAI/bge-m3（輸出 1024 維向量）
- **分類模型**：BiLSTM
- **優化器**：AdamW
- **損失函數**：CrossEntropyLoss
- **評估指標**：Accuracy、Precision、Recall、F1-score

## 5. 依賴項
請先安裝所需的 Python 套件，可使用以下指令：
```
pip install -r requirements.txt
```

本專案涵蓋了從文本處理到模型應用的完整流程，適用於新聞文本的自動摘要與分類，未來可以進一步優化並應用於真實業務環境。

