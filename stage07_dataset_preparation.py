import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

class NewsDataset(Dataset):
    """
    自訂 Dataset 類別，處理 Token 化後的新聞文本
    - 讀取 .pt 格式的 Tokenized 檔案
    - 依據資料夾名稱建立標籤
    - 可被 PyTorch DataLoader 直接讀取
    """
    def __init__(self, tokenized_dir:str):
        super().__init__()
        self.paths = list(Path(tokenized_dir).glob('**/*.pt'))
        self.classes, self.classes_idx = self._find_classes(tokenized_dir)
        print(self.classes)
        print(self.classes_idx)

    def _find_classes(self, tokenized_dir:str):
        entries = [entry for entry in os.scandir(tokenized_dir) if entry.is_dir()]
        if not entries:
            raise FileNotFoundError(f"Couldn't find any class in {tokenized_dir}")
        classes = [entry.name for entry in entries]
        classes_idx = {entry: i for i, entry in enumerate(classes)}
        return classes, classes_idx

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index:int):
        class_name = self.paths[index].parent.name
        label = torch.tensor(self.classes_idx[class_name], dtype=torch.long) 
        tokenized_data = torch.load(self.paths[index], weights_only=True)
        return tokenized_data, label

if __name__ == "__main__": 
    news_data_path = 'data/traditional_train_clean_summary_200'
    dataset = NewsDataset(tokenized_dir=news_data_path)
    torch.save(dataset, 'data/train_dataset.pt')
    print(len(dataset))

    news_data_path = 'data/traditional_test_clean_summary_100'
    dataset = NewsDataset(tokenized_dir=news_data_path)
    torch.save(dataset, 'data/test_dataset.pt')
    print(len(dataset))






