import torch
from dataset.dataloader import dataloader
from model.nn import NeuralNetwork
from model.loss import CeLoss
from flow.flow import train, evaluate
# from torch.optim.lr_scheduler import StepLR

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 數據路徑
file_path = "data/train.csv"
test_features_path = "data/test.csv"
test_labels_path = "data/gender_submission.csv"

epochs = 50
batch_size = 16
input_dim = 5
num_classes = 2
learning_rate = 1e-3
model_path = "savemodel"

train_loader, valid_loader = dataloader(batch_size, file_path)
test_loader = dataloader(batch_size, test_features_path, test_labels_path, mode="test")

model = NeuralNetwork(input_dim, num_classes).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # 每 20 個 epoch，學習率變為原來的 50%（gamma=0.5）
# scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

criterion = CeLoss().to(DEVICE)

if __name__ == "__main__":
    
    best = 0
    train_losses = []
    valid_losses = []

    train_mtx = []
    valid_mtx = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_accuracy = train(now_ep=epoch,
                                          model=model,
                                          optimizer=optimizer,
                                          dataloader=train_loader,
                                          criterion=criterion,
                                          DEVICE=DEVICE)

        valid_loss, valid_accuracy = evaluate(mode="validation",
                                             model=model,
                                             dataloader=valid_loader,
                                             criterion=criterion,
                                             DEVICE=DEVICE)
        

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_mtx.append(train_accuracy)
        valid_mtx.append(valid_accuracy)

        # # 讓學習率根據 scheduler 變小
        # scheduler.step()

         # 顯示當前 epoch 和學習率
        # print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]}")


        if max(valid_mtx) >= best:
            best = max(valid_mtx) 
            # **只存權重**
            torch.save(model.state_dict(), model_path + "/best_model_weights.pth")
            # **存整個模型**
            torch.save(model, model_path + "/best_model_full.pth")
            print(f"Best model updated with Validation Accuracy: {best:.4f}")


        
print("\nTraining complete. Loading best model (weights only) for testing...")

# **重新初始化模型**
model = NeuralNetwork(input_dim, num_classes).to(DEVICE)

# **載入 "權重"**
model.load_state_dict(torch.load(model_path + "/best_model_weights.pth",weights_only=True))
model.eval()  # 設定為 evaluation 模式

# **執行測試**
test_loss, test_accuracy = evaluate(
    mode="test",
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    DEVICE=DEVICE
)

print(f"\nTest Results (Weights Only) -> Test Accuracy: {test_accuracy:.4f}")



# from sklearn.metrics import classification_report, confusion_matrix

# y_true, y_pred = [], []

# for inputs, labels in test_loader:
#     outputs = model(inputs.to(DEVICE))
#     _, preds = torch.max(outputs, 1)
#     y_true.extend(labels.numpy())
#     y_pred.extend(preds.cpu().numpy())

# print(classification_report(y_true, y_pred))
# print(confusion_matrix(y_true, y_pred))
