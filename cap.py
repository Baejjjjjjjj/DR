import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("******" , DEVICE)

# 데이터셋 정의
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 2]  # 'path' 열
        image = Image.open(img_path)
        label = int(self.dataframe.iloc[idx, 1])  # 'level' 열

        if self.transform:
            image = self.transform(image)

        return image, label

# 이미지 변환
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# CSV 파일 로드
print("Loading CSV file...")
trainLabels = pd.read_csv("./trainLabels.csv")
print(f"CSV 파일 로드 성공: {trainLabels.shape[0]}개의 데이터 로드됨")

# 이미지 경로 생성
base_image_dir = os.path.join('.', 'train')
print("base_이미지 경로 : " + base_image_dir)
trainLabels['path'] = trainLabels['image'].map(lambda x: os.path.join(base_image_dir, f'{x}.jpeg'))
print(f"이미지 경로 생성 완료, 경로 예시: {trainLabels['path'].iloc[0]}")

# 파일이 존재하는지 확인
trainLabels = trainLabels[trainLabels['path'].map(os.path.exists)]
if trainLabels.shape[0] == 0:
    raise ValueError("No valid images found. Check file paths.")
print(f"유효한 이미지 파일: {trainLabels.shape[0]}개")

# 레벨을 문자열로 변환
trainLabels['level'] = trainLabels['level'].astype(str)
print("레벨을 문자열로 변환 완료")

# train_test_split 호출 전 데이터 출력
print(f"Train Labels 데이터 미리보기:\n{trainLabels.head()}")

# 데이터 분할
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(trainLabels, test_size=0.2, stratify=trainLabels['level'])
train_df = train_df.sample(n=3000, random_state=42)
val_df = val_df.sample(n=1000, random_state=42)
print("train_test_split 성공")

invalid_paths = [path for path in trainLabels['path'] if not os.path.exists(path)]
print(f"존재하지 않는 경로 개수: {len(invalid_paths)}")
print("존재하지 않는 경로들:", invalid_paths)


print("Train DataFrame 미리보기:")
print(train_df.head())
print(f"Train DataFrame 크기: {train_df.shape}")

train_dataset = DiabeticRetinopathyDataset(train_df, transform=transform)
val_dataset = DiabeticRetinopathyDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train Dataset 크기: {len(train_dataset)}")

for data in train_loader:
    print(f"data의 타입: {type(data)}, 데이터 길이: {len(data)}")
    break

for inputs, labels in train_loader:
    print(f"입력 데이터 타입: {type(inputs)}, 레이블 데이터 타입: {type(labels)}")
    break  # 첫 번째 배치만 출력

for inputs, labels in train_loader:
    print(f"입력 데이터 크기: {inputs.shape}, 레이블 크기: {labels.shape}")
    break  # 첫 번째 배치만 출력

# EfficientNet 모델 불러오기
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 함수
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=EPOCHS):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation
        val_acc = evaluate_model(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_efficientnet.pth")

    print(f"Best validation accuracy: {best_acc:.4f}")

# 평가 함수
def evaluate_model(model, val_loader):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    val_acc = running_corrects.double() / len(val_loader.dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")
    return val_acc

# 학습 실행
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=EPOCHS)

# 혼동 행렬 및 Classification Report 출력
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 모델 평가 및 시각화
model.load_state_dict(torch.load("best_efficientnet.pth"))

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 혼동 행렬 및 classification report
cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix')
print(cm)

target_names = [str(i) for i in range(NUM_CLASSES)]
print('Classification Report')
print(classification_report(all_labels, all_preds, target_names=target_names))

# 혼동 행렬 시각화
plot_confusion_matrix(cm, target_names)
