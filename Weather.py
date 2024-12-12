import requests
import json
import pandas as pd

# API 엔드포인트 설정
url = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# API 호출 파라미터
params = {
    'serviceKey': 'xQWkDcQS71sHlEunWkffKRkPICXYsCy0YxlYfdX/ABjb+BHWkBFnqW8Kk0p1+gWIYxEB8ZQR1sxGsFQpXBVs3A==',
    'pageNo': '1',
    'numOfRows': '500',
    'dataType': 'JSON',
    'dataCd': 'ASOS',
    'dateCd': 'DAY',
    'startDt': '20211215',  # 시작 날짜
    'endDt': '20241009',  # 종료 날짜
}

# 'stnIds'를 108부터 295까지의 숫자로 설정
stn_ids = [str(i) for i in range(108, 296)]  # station IDs: 108, 109, ..., 295

weather_data = []

# 모든 stnId에 대해 반복해서 데이터를 수집
for stn_id in stn_ids:
    params['stnIds'] = stn_id  # stnId 설정
    response = requests.get(url, params=params)

    # 응답 상태 코드 확인
    if response.status_code != 200:
        print(f"Request failed for station {stn_id} with status code {response.status_code}")
        continue  # 다음 stnId로 넘어감

    try:
        data = response.json()

    except json.JSONDecodeError:
        print(f"Error decoding JSON for station {stn_id}")
        continue  # JSON 디코딩 실패 시 다음 stnId로 넘어감

    print("******")
    # 데이터 확인
    if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body'] and 'item' in \
            data['response']['body']['items']:
        for item in data['response']['body']['items']['item']:
            weather_data.append({
                'stnId': item.get('stnId', None),
                'stnNm': item.get('stnNm', None),
                'tm': item.get('tm', None),
                'avgTa': item.get('avgTa', None),
                'minTa': item.get('minTa', None),
                'minTaHrmt': item.get('minTaHrmt', None),
                'maxTa': item.get('maxTa', None),
                'maxTaHrmt': item.get('maxTaHrmt', None),
                'sumRnDur': item.get('sumRnDur', None),
                'mi10MaxRn': item.get('mi10MaxRn', None),
                'mi10MaxRnHrmt': item.get('mi10MaxRnHrmt', None),
                'hr1MaxRn': item.get('hr1MaxRn', None),
                'hr1MaxRnHrmt': item.get('hr1MaxRnHrmt', None),
                'sumRn': item.get('sumRn', None),
                'maxInsWs': item.get('maxInsWs', None),
                'maxInsWsWd': item.get('maxInsWsWd', None),
                'maxInsWsHrmt': item.get('maxInsWsHrmt', None),
                'maxWs': item.get('maxWs', None),
                'maxWsWd': item.get('maxWsWd', None),
                'maxWsHrmt': item.get('maxWsHrmt', None),
                'avgWs': item.get('avgWs', None),
                'hr24SumRws': item.get('hr24SumRws', None),
                'maxWd': item.get('maxWd', None),
                'avgTd': item.get('avgTd', None),
                'minRhm': item.get('minRhm', None),
                'minRhmHrmt': item.get('minRhmHrmt', None),
                'avgRhm': item.get('avgRhm', None),
                'avgPv': item.get('avgPv', None),
                'avgPa': item.get('avgPa', None),
                'maxPs': item.get('maxPs', None),
                'maxPsHrmt': item.get('maxPsHrmt', None),
                'minPs': item.get('minPs', None),
                'minPsHrmt': item.get('minPsHrmt', None),
                'avgPs': item.get('avgPs', None),
                'ssDur': item.get('ssDur', None),
                'sumSsHr': item.get('sumSsHr', None),
                'hr1MaxIcsrHrmt': item.get('hr1MaxIcsrHrmt', None),
                'hr1MaxIcsr': item.get('hr1MaxIcsr', None),
                'sumGsr': item.get('sumGsr', None),
                'ddMefs': item.get('ddMefs', None),
                'ddMefsHrmt': item.get('ddMefsHrmt', None),
                'ddMes': item.get('ddMes', None),
                'ddMesHrmt': item.get('ddMesHrmt', None),
                'sumDpthFhsc': item.get('sumDpthFhsc', None),
                'avgTca': item.get('avgTca', None),
                'avgLmac': item.get('avgLmac', None),
                'avgTs': item.get('avgTs', None),
                'minTg': item.get('minTg', None),
                'avgCm5Te': item.get('avgCm5Te', None),
                'avgCm10Te': item.get('avgCm10Te', None),
                'avgCm20Te': item.get('avgCm20Te', None),
                'avgCm30Te': item.get('avgCm30Te', None),
                'avgM05Te': item.get('avgM05Te', None),
                'avgM10Te': item.get('avgM10Te', None),
                'avgM15Te': item.get('avgM15Te', None),
                'avgM30Te': item.get('avgM30Te', None),
                'avgM50Te': item.get('avgM50Te', None),
                'sumLrgEv': item.get('sumLrgEv', None),
                'sumSmlEv': item.get('sumSmlEv', None),
                'n99Rn': item.get('n99Rn', None),
                'iscs': item.get('iscs', None),
                'sumFogDur': item.get('sumFogDur', None),
            })

# 데이터프레임으로 변환
df = pd.DataFrame(weather_data)

print(df.columns)
print("***********")
# pivot_table 생성할 때 없는 열을 처리하도록 에러 핸들링
try:
    df_pivot = df.pivot_table(
        index='tm',
        columns='stnNm',
        values=[  # 필요 데이터 목록
            'avgTa', 'minTa', 'maxTa', 'minTaHrmt', 'maxTaHrmt',
            'sumRnDur', 'mi10MaxRn', 'mi10MaxRnHrmt', 'hr1MaxRn', 'hr1MaxRnHrmt',
            'sumRn', 'maxInsWs', 'maxInsWsWd', 'maxInsWsHrmt', 'maxWs', 'maxWsWd', 'maxWsHrmt',
            'avgWs', 'hr24SumRws', 'maxWd', 'avgTd', 'minRhm', 'minRhmHrmt', 'avgRhm',
            'avgPv', 'avgPa', 'maxPs', 'maxPsHrmt', 'minPs', 'minPsHrmt', 'avgPs',
            'ssDur', 'sumSsHr', 'hr1MaxIcsrHrmt', 'hr1MaxIcsr', 'sumGsr',
            'ddMefs', 'ddMefsHrmt', 'ddMes', 'ddMesHrmt', 'sumDpthFhsc',
            'avgTca', 'avgLmac', 'avgTs', 'minTg', 'avgCm5Te', 'avgCm10Te',
            'avgCm20Te', 'avgCm30Te', 'avgM05Te', 'avgM10Te', 'avgM15Te',
            'avgM30Te', 'avgM50Te', 'sumLrgEv', 'sumSmlEv', 'n99Rn', 'iscs', 'sumFogDur'
        ],
        aggfunc='first'
    )
    print(df_pivot)
except KeyError as e:
    print(f"Column {e} does not exist in the dataframe.")
print("index reset before")


# 인덱스를 리셋하고 결과를 CSV로 저장
df_pivot.reset_index(inplace=True)
df_pivot.to_csv('weather_data.csv', index=False)

print("Data saved to weather_data.csv")
