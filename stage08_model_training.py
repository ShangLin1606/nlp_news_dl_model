import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import time
from stage07_dataset_preparation import NewsDataset
import matplotlib.pyplot as plt
import pandas as pd
from torchinfo import summary

def elapsed_time(start, end):
    elapsed_time = end - start
    print(f"Program execution time: {elapsed_time:.5f} sec")
    return round(elapsed_time, 2)

# 使用 BAAI/bge-m3 embedding 的新聞分類器
class NewsClassifier(nn.Module):
    def __init__(self, output_dim):
        super(NewsClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        self.embedding_model = AutoModel.from_pretrained('BAAI/bge-m3')
        self.lstm = nn.LSTM(self.embedding_model.config.hidden_size, 256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256 * 2, output_dim)

    def forward(self, x, attention_mask=None):
        with torch.no_grad():
            if attention_mask is None:
                attention_mask = (x != self.tokenizer.pad_token_id).long()  # 假設 pad_token_id = 0
            outputs = self.embedding_model(input_ids=x, attention_mask=attention_mask)
            x = outputs.last_hidden_state
        x, (h_n, c_n) = self.lstm(x)
        h_forward = self.dropout(h_n[-2])  # 前向LSTM最後一層 + Dropout
        h_backward = self.dropout(h_n[-1])  # 後向LSTM最後一層 + Dropout
        x = torch.cat((h_forward, h_backward), dim=1)  # 拼接
        x = self.fc(x)
        return x


# 訓練與測試函數
def evaluate(model, dataloader, loss_fn, device, target_names=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    if target_names == None:
        print(classification_report(all_labels, all_preds, digits=4))
    else:
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    return running_loss / len(dataloader), acc, precision, recall, f1

if __name__ == "__main__":
    print(NewsClassifier(output_dim=20))



    if False:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data = torch.load('data/train_dataset.pt', weights_only=False)
        test_data = torch.load('data/test_dataset.pt', weights_only=False)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=os.cpu_count()-1, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=32, num_workers=os.cpu_count()-1, pin_memory=True)

        model = NewsClassifier(output_dim=20).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)

        epochs = 20
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []
        results = []


        start = time.time()
        for epoch in tqdm(range(epochs)):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, loss_fn, device, train_data.classes)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            scheduler.step(test_loss)
            
            print(f"epoch [{epoch+1}/{epochs}] train Loss: {train_loss:.4f} train Acc: {train_acc:.2f}% | test Loss: {test_loss:.4f} test Acc: {test_acc:.2f}% test precision: {test_precision:.2f} test recall: {test_recall:.2f} test f1: {test_f1:.2f}")
            results.append({
                "Epoch": epoch + 1,
                "Train Loss": round(train_loss,2),
                "Train Accuracy": round(train_acc,2),
                "Test Loss": round(test_loss,2),
                "Test Accuracy": round(test_acc,2),
                "Test Precision": round(test_precision,2),
                "Test Recall": round(test_recall,2),
                "Test F1 Score": round(test_f1,2)
            })

        print("訓練完成！")
        end = time.time()
        execution_time = elapsed_time(start, end)

        torch.save(model.state_dict(), 'classifier_BGE-m3_BiLSTM256_ep20_lr1e-3_plateau_adamw_do3e-1_wd1e-2.pth.pth')
        print("模型參數已儲存！")
        df_results = pd.DataFrame(results)

        df_results.to_csv("training_results.csv", index=False)
        print("所有訓練結果已儲存至 `training_results.csv`！")

        # 繪製 Loss & Accuracy 曲線
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training & Test Loss")

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Train Acc")
        plt.plot(test_accuracies, label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Training & Test Accuracy")
        plt.show()

    
    
    
    












































# # 使用 BAAI embedding 模型的新聞分類器
# class NewsClassifier(nn.Module):
#     def __init__(self, output_dim):
#         super(NewsClassifier, self).__init__()
#         self.embedding_model = AutoModel.from_pretrained('BAAI/bge-m3')
#         self.lstm = nn.LSTM(self.embedding_model.config.hidden_size, 256, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(256 * 2, output_dim)

#     def forward(self, x):
#         with torch.no_grad():
#             x = self.embedding_model(x).last_hidden_state  # 使用 BAAI embedding
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         return x

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_list = torch.load('data/test_train_dataset.pt', weights_only=True)
#     dataloader = DataLoader(data_list, batch_size=32, shuffle=True, num_workers=os.cpu_count()-1)

#     output_dim = 2  # 類別數量
#     model = NewsClassifier(output_dim).to(device)

#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=0.001)

#     epochs = 5
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = loss_fn(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(dataloader):.4f} Accuracy: {100 * correct/total:.2f}%")

#     print("訓練完成！")



