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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # 회귀 문제에서는 단일 값을 예측하므로 NUM_CLASSES는 필요하지 않음
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("****** 실행 디바이스:", DEVICE)

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

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# CSV 파일 로드
print("CSV 파일 로드 중...")
trainLabels = pd.read_csv("./trainLabels.csv")
print(f"CSV 파일 로드 성공: {trainLabels.shape[0]}개의 데이터 로드됨")

# 이미지 경로 생성
base_image_dir = os.path.join('.', 'train')
trainLabels['path'] = trainLabels['image'].map(lambda x: os.path.join(base_image_dir, f'{x}.jpeg'))
trainLabels = trainLabels[trainLabels['path'].map(os.path.exists)]

# 데이터 확인 출력
if trainLabels.shape[0] == 0:
    raise ValueError("유효한 이미지 파일이 없습니다. 파일 경로를 확인하세요.")
print(f"유효한 이미지 파일 수: {trainLabels.shape[0]}개")

# 데이터 분할
train_df, val_df = train_test_split(trainLabels, test_size=0.2, stratify=trainLabels['level'])
train_df = train_df.sample(n=3000, random_state=42)
val_df = val_df.sample(n=600, random_state=42)
print("train_test_split 성공")

# DataLoader 설정
train_dataset = DiabeticRetinopathyDataset(train_df, transform=transform)
val_dataset = DiabeticRetinopathyDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train Dataset 크기: {len(train_dataset)}개")
print(f"Validation Dataset 크기: {len(val_dataset)}개")

# EfficientNet 모델 설정
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
model = model.to(DEVICE)

# 손실 함수 및 옵티마이저 정의
criterion = nn.SmoothL1Loss()  # 회귀를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 함수
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=EPOCHS,
                checkpoint_path="checkpoint_efficientnet.pth"):
    best_loss = float('inf')
    start_epoch = 0

    # Checkpoint 로드 중
    # Checkpoint 로드 중
    # Checkpoint 로드 중
    if os.path.exists(checkpoint_path):
        print("Checkpoint 불러오는 중...")

        checkpoint = torch.load(checkpoint_path, weights_only=True)  # `weights_only=True` 추가

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)  # 'epoch' 키가 없으면 0으로 설정
        best_loss = checkpoint.get('best_loss', float('inf'))  # 'best_loss' 키가 없으면 inf로 설정

        print(f"Checkpoint 로드 성공: {start_epoch} 에폭부터 시작, 초기 best_loss: {best_loss}")
    else:
        best_loss = float('inf')  # 새로운 학습 시작 시 초기 best_loss 값을 무한대로 설정

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(DEVICE)

    for epoch in range(start_epoch, num_epochs):
        print("*****************************")
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            # 손실 계산
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation Loss 확인
        val_loss = evaluate_model(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_efficientnet.pth")
            print(f"새로운 최고 손실: {best_loss:.4f}, 모델 저장")

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)

    print(f"Best validation loss: {best_loss:.4f}")

# 평가 함수
def evaluate_model(model, val_loader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float()
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

# 학습 실행
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=EPOCHS)

# 예측값 및 실제값 수집
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# 회귀 평가 지표 출력 (예: MAE, MSE)
mae = mean_absolute_error(all_labels, all_preds)
mse = mean_squared_error(all_labels, all_preds)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")