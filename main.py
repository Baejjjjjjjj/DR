from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import time
import json

# 크롬 드라이버 경로 설정
chrome_driver_path = './chromedriver.exe'  # 자신의 chromedriver 경로로 변경
service = Service(chrome_driver_path)

# WebDriver 설정
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# 실행 시간 측정 시간
start_time = time.time()

# 웹 페이지 열기
url = 'https://www.weather.go.kr/w/weather/commentary.do'  # 방문할 웹페이지의 URL로 변경
driver.get(url)

# 페이지가 완전히 로딩될 때까지 대기 (필요 시 추가 대기 시간 조정)
time.sleep(3)

# 추출된 텍스트들을 저장할 리스트
all_texts = []

for i in range(1,680):
    pages = driver.find_element(By.CLASS_NAME, "page-num.page")
    page_buttons = pages.find_elements(By.TAG_NAME, "button")

    next_page_button = page_buttons[i%8]

    if not i%8: continue

    if i%8 == 7:
        next_page_button.click()
        time.sleep(3)
        continue
    next_page_button.click()
    time.sleep(3)

    ulitem = driver.find_element(By.ID, "weather-on-list")

    liitems = ulitem.find_elements(By.TAG_NAME, "li")

    # 각 li 내의 버튼 클릭 후 텍스트 추출
    for index, li_item in enumerate(liitems):
        try:

            # 특정 버튼을 찾아서 클릭 (예: 버튼의 id, 클래스명, 태그 등을 활용)
            # xpath_button = '//*[@id="weather-on-list"]/li[1]/div[1]/div[2]'  # 실제 버튼의 XPath로 변경
            button = li_item.find_element(By.CSS_SELECTOR, "div.move-list-btn.accordion-tit")
            button.click()

            # 클릭 후 필요한 시간이 있을 경우 대기
            time.sleep(1)

            # 날짜 추출
            date = li_item.find_element(By.CSS_SELECTOR, "div.move-list-btn.accordion-tit > div.right-btn > div > span").text

            # 텍스트 추출 (예: 특정 요소의 텍스트를 가져옴)
            # xpath_text = '//*[@id="weather-on-list"]/li[1]/div[2]/div[2]/div/div/div/div/p'  # 실제 텍스트 요소의 XPath로 변경
            text_element = li_item.find_element(By.CSS_SELECTOR, "div.move-list-con.accordion-con > div.right-con > div > div > div > div > p")
            extracted_text = text_element.text.replace('\n', ' ')

            json_file_path = "weather_script_data.json"
            with open(json_file_path, 'a', encoding='utf-8') as f:
                f.write('{' + f'"date": "{date}",\n"txt" : "{extracted_text}"' + "}, \n")
            # 텍스트를 리스트에 저장
            all_texts.append({"date": date, "txt" : extracted_text})


            # 추가 작업이 있을 경우 추가 (예: 다시 원래 페이지로 돌아가기 등)
        except Exception as e:
            print(f"Error processing li item {index}: {str(e)}")
            continue  # 에러 발생 시 다음 li로 넘어감

    # 저장 후 리스트 초기화
    all_texts.clear()


# 크롤링 완료 후 브라우저 닫기
# driver.quit()
json_file_path = "weather_script_data.json"
with open(json_file_path, 'a', encoding='utf-8') as f:
    f.write(']')


end_time = time.time()
print(f"프로그램 실행 시간 : {end_time - start_time}")

time.sleep(1000)