from datetime import datetime

from tables.tests.test_queries import type_info
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import os
import json
import torch
import numpy as np
import sys

# 환경설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
torch.cuda.empty_cache()
# GPT 모델 및 토크나이저 로드
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "./my_trained_model"  # Directory where the model is saved
if os.path.exists(model_dir):
    print("Loading saved model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
else:
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")

# padding 토큰 추가
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token':'[SEP]'})
model.resize_token_embeddings(len(tokenizer))

# CSV 파일에서 날씨 데이터 로드
weather_data = pd.read_csv('weather_data_1112.csv', encoding='utf-8')

# JSON 파일에서 뉴스 데이터 로드
with open('weather_script_data.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 날짜 형식을 'YYYY-MM-DD'로 맞추기 위한 함수
def format_date(date_str):
    if pd.isna(date_str):
        return None
    if isinstance(date_str, str) and len(date_str) == 10:
        return date_str
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    except ValueError:
        return None

# 날짜를 기준으로 날씨 데이터와 뉴스 텍스트를 페어링
paired_data = []
for i, row in weather_data.iterrows():
    date = row['tm']
    formatted_date = format_date(date)

    if formatted_date is None:
        continue

    weather_info = {col: row[col] for col in weather_data.columns if col != 'tm' and pd.notna(row[col])}
    news_item = next((item for item in news_data if format_date(item['date']) == formatted_date), None)

    if news_item:
        paired_data.append({
            'date': formatted_date,
            'weather': weather_info,
            'news': news_item['txt']
        })

# 프롬프트 텍스트 추가
def create_prompted_text(index, total_parts, date, weather, news=None):
    weather = json.dumps(weather, ensure_ascii=False)
    weather_list = []
    news_list = []
    if news:
        weather = date+ '날씨 정보: ' + weather + ' [SEP] 뉴스 기사: '+ news
        #print("토큰크기",tokenizer(weather, return_tensors='pt', truncation=True)['input_ids'].shape)
        #print(weather)
        wl = len(weather)
        #print(wl)
        for i in range(0,wl,wl//3):
            weather_list.append(weather[i:i+(wl//2)])

        #print(weather_list)
        return weather_list + news_list
    else:
        weather = date+ '날씨 정보: ' + weather + '일때, [SEP] 뉴스 기사만 말해줘 '
        wl = len(weather)

       # print(weather_list)

        return [weather]

# 데이터셋 준비
total_parts = 10  # 각 입력이 10개 조각으로 구성되도록 설정
sample_size = 200  # 학습 및 테스트 데이터셋 각각 100개씩 사용
train_data_set, test_data_set = [], []


for idx, item in enumerate(paired_data):  # 전체 paired_data에서 200개만 사용
    #print(idx)
    #print(item['weather'])
    if idx < sample_size:  # 테스트용 데이터셋 구성
        test_data_set.extend(create_prompted_text(1, total_parts, item['date'], item['weather']))
    else:  # 학습용 데이터셋 구성
        train_data_set.extend(create_prompted_text(1, total_parts, item['date'], item['weather'], item['news']))


# 데이터 토큰화 및 1024 토큰 이하로 분할
def chunk_data(data_set):
    tokenized_chunks = []
    for data in data_set:
        #print("data", data)
        tokenized_data = tokenizer(data, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        input_ids = tokenized_data['input_ids'].squeeze().numpy().astype(np.float32)
        attention_mask = tokenized_data['attention_mask'].squeeze().numpy().astype(np.float32)

        tokenized_chunks.append([input_ids, attention_mask])

    return tokenized_chunks

train_chunks = torch.tensor(chunk_data(train_data_set)).long()
test_chunks = torch.tensor(chunk_data(test_data_set)).long()

# Dataset 객체로 변환
train_dataset = Dataset.from_dict({
    'input_ids': [list(chunk[0]) for chunk in train_chunks],
    'attention_mask': [list(chunk[1]) for chunk in train_chunks],
    'labels': [list(chunk[0]) for chunk in train_chunks]
})

test_dataset = Dataset.from_dict({
    'input_ids': [list(chunk[0]) for chunk in test_chunks],
    'attention_mask': [list(chunk[1]) for chunk in test_chunks],
    'labels': [list(chunk[0]) for chunk in test_chunks]
})

# 학습 설정 및 Trainer 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

torch.cuda.empty_cache()

# 모델 학습
trainer.train()

# 모델 저장
output_dir = "./my_trained_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 테스트 데이터셋 평가
eval_results = trainer.evaluate()
print(f"평가 결과: {eval_results}")

# 테스트 데이터로 생성 결과 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
# SmoothingFunction: BLEU 점수가 너무 낮게 나오는 것을 방지
smooth = SmoothingFunction().method4

# BLEU 점수 저장 변수
bleu_scores = []

def generate_weather_news(input_ids):

    generated_ids = model.generate(input_ids,  max_length=1000, num_return_sequences=1, top_p=0.9, top_k=50, repetition_penalty=1.2)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

# 첫 번째 뉴스 기사 요약하는 함수
def summarize_text(text):
    input_ids = tokenizer.encode("뒤 내용을 100자 이내로 요약", text, return_tensors='pt').to(device)
    summary_ids = model.generate(input_ids, max_length=1000, top_p=0.9, top_k=50, repetition_penalty=1.2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("summarize_text: ", summary)
    return summary
def generate_followup_news(input_ids, first_article_summary):
    # 첫 번째 뉴스 기사 요약을 이어서 생성할 텍스트로 추가
    input_text = f"[SEP] 첫 번째 뉴스 기사 요약: {first_article_summary} [SEP] 이어서 뉴스 기사를 작성, '요'체를 사용하지 말것 "

    # input_ids에 추가 텍스트를 연결하여 전체 입력을 준비
    input_ids_with_summary = input_ids.tolist()[0] + tokenizer.encode(input_text, add_special_tokens=False)

    # 모델에 입력하여 이어서 텍스트 생성
    generated_ids = model.generate(torch.tensor([input_ids_with_summary]).to(device), max_length=1024, num_return_sequences=1,
                                   top_p=0.9, top_k=50, repetition_penalty=1.2)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# 테스트 데이터를 바탕으로 예측 생성 및 BLEU 평가
for j in range(len(test_dataset)):  # 테스트 데이터셋에서 각 샘플 처리
    input_ids = torch.tensor(test_dataset[j]['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(test_dataset[j]['attention_mask']).unsqueeze(0).to(device)

    # 모델 예측 생성
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=500,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2
    )

    summarizeText=""
    generated_text=""
    keyword ="뉴스"
    if(j%2!=0):
        generated_first = generate_weather_news(input_ids)
        if keyword in generated_first:
            start = generated_first.find(keyword) + len(keyword)
            generated_first = generated_first[start:].strip()
            print("가공된 generated_first:" + generated_first)
            summarize_text(generated_first)
            continue
        else:
            j+=1
            continue
    else:
        generated_text = generate_followup_news(input_ids, summarizeText)
        print("최종 뉴스:", generated_text)
    # 생성된 텍스트와 참조 텍스트
    #generated_text = tokenizer.decode(generated_texts, skip_special_tokens=True)
    reference_text = paired_data[j]['news']  # paired_data에서 참조 뉴스 스크립트

    # BLEU 평가
    if reference_text and '뉴스' in generated_text:
        generated_content = generated_text.split("뉴스")[1].strip()  # 생성된 뉴스 스크립트만 추출
        reference_tokens = reference_text.split()  # 참조 텍스트 토큰화
        generated_tokens = generated_content.split()  # 생성된 텍스트 토큰화

        # BLEU 점수 계산
        bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu_score)

        # 결과 출력
        print(f"Sample {j + 1}:")
        print(f"Reference: {reference_text}")
        print(f"Generated: {generated_content}")
        print(f"BLEU Score: {bleu_score:.4f}\n")
    else:
        print(f"Sample {j + 1}: 뉴스 기사가 생성되지 않았습니다.")

# 평균 BLEU 점수 출력
if bleu_scores:
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"\nAverage BLEU Score: {avg_bleu_score:.4f}")
else:
    print("\nNo valid BLEU scores calculated.")


def generate_weather_news(weather_data):
    """
    주어진 날씨 데이터를 기반으로 뉴스 기사를 생성하는 함수.

    Args:
        weather_data (str): 날씨 정보 문자열.

    Returns:
        str: 생성된 뉴스 기사.
    """
    try:
        # 입력 텍스트를 토큰화
        input_ids = tokenizer(weather_data, return_tensors='pt', truncation=True, max_length=512)['input_ids']
        input_ids = input_ids.to(device)

        # 모델을 사용하여 텍스트 생성
        generated_ids = model.generate(
            input_ids,
            max_length=500,  # 생성할 최대 토큰 수
            num_return_sequences=1,  # 생성할 텍스트 개수
            top_k=50,  # 샘플링을 제한할 상위 K개
            top_p=0.9,  # 누적 확률 기반의 샘플링
            repetition_penalty=1.2,  # 반복 방지
            pad_token_id=tokenizer.pad_token_id
        )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"뉴스 생성 중 오류 발생: {e}")
        return "뉴스를 생성할 수 없습니다."