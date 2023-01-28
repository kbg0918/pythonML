# -*- coding: utf-8 -*-


#데이터 처리를 위한 라이브러리
import pandas as pd

#웹 페이지의 html 코드를 수동으로 사용할 수 있게 해주는 라이브러리
from bs4 import BeautifulSoup

#코드만 작성하면 자동화하여 데이터를 뽑아올 수 있게 해주는 라이브러리
from selenium import webdriver
#Select라는 html코드에 접근할 수 있게 해주는 라이브러리
from selenium.webdriver.support.ui import Select



# import requests
# import unicodedata
# from urllib.request import urlopen

# import selenium
# from selenium.webdriver import ActionChains

# from selenium.webdriver.support import expected_conditions as EC

# from selenium.webdriver.support.ui import WebDriverWait

def crawling():
    

    #selenium 준비
    
    #데이터를 추출할 웹 페이지의 uri입니다.
    uri = 'http://www.weatheri.co.kr/bygone/bygone01.php'
    #크롬에서 저희가 보는 모든 데이터를 가져오려면 chromedriver을 설치해야합니다.
    driver = webdriver.Chrome(executable_path='chromedriver')
    driver.get(uri)
    #페이지 로딩이 완료될 때까지 기다리는 코드입니다.
    driver.implicitly_wait(time_to_wait=5)
    
    
    #Select에 name 변수에 들어있는 s_year를 찾아 1970으로 지정
    select1 = Select(driver.find_element_by_name("s_year"))
    #Select>option에 있는 value값을 1970으로 지정
    select1.select_by_value("1970")
    #1970년 1월 1일부터 2020년 12월 31일까지의 데이터를 찾기위해 위와 같은 작업을 합니다.
    select2 = Select(driver.find_element_by_name("s_month"))
    select2.select_by_value("1")
    
    select3 = Select(driver.find_element_by_name("s_day"))
    select3.select_by_value("1")
            
    select4 = Select(driver.find_element_by_name("e_year"))
    select4.select_by_value("2020")
    
    select5 = Select(driver.find_element_by_name("e_month"))
    select5.select_by_value("12")
    
    select6 = Select(driver.find_element_by_name("e_day"))
    select6.select_by_value("31")
    #검색하기 버튼을 동적으로 눌러주기 위해 xpath값을 설정합니다.
    serch = driver.find_element_by_xpath('/html/body/table[2]/tbody/tr[3]/td[2]/table/tbody/tr[2]/td/img')
    #검색하기 클릭
    serch.click()
    
    
    temp = []
    
    
    #iframe에 해당하는 코드를 가져오기 위해 switch해주어야 한다.
    driver.switch_to_frame('kako')      
    
    
    
    #길이는 뽑을 데이터의 년도+1 만큼        
    for i in range(1970,2021):
        #뽑을 데이터의 달+1 만큼
        for j in range(1,13):
            #1989년과 2007년은 데이터가 비어있어서 break 처리 해주었습니다.
            if i == 1989 or i == 2007:
                break
            else:
                select = Select(driver.find_element_by_name("start"))
                #i와 j값을 받아 동적으로 년도를 설정하기 위해 문자열로 바꿔주는 str()을 사용했습니다.
                if j < 10:
                    select.select_by_value(str(i)+"-0"+str(j)+"-01")
                elif j >= 10:
                    select.select_by_value(str(i)+"-"+str(j)+"-01")
                
                
                # 드롭다운 메뉴 클릭
                inj = driver.find_element_by_xpath('/html/body/table/tbody/tr/td/form/table[1]/tbody/tr/td[2]/img')
                inj.click()
                
                # #BeautifulSoup를 사용하여 데이터 가져오기
                
                #BeautifulSoup를 사용하기 위해 html 변수에 page 소스 저장
                html = driver.page_source
                soup=BeautifulSoup(html, 'html.parser')
                
                #tbody의 xpath값을 저장하여 tbody안에 있는 tag 사용
                t = driver.find_element_by_xpath('/html/body/table/tbody/tr/td/form/table[2]/tbody')
                
                #tr의 0번째와 1번째를 무시하기 위해 tr을 list로 바꾸어주었습니다.
                #t(tbody)안에 있는 tr태그를 사용하기 위한 코드입니다.
                trs = list(t.find_elements_by_tag_name('tr'))
                
                #tr의 길이만큼 반복문을 돕니다.
                for k in range(len(trs)):
                    #tr의 0번째와 1번째 값이 비어있지 않을 때 (조건문)
                    if k > 1:
                        #trs[k]요소마다 접근하여 tr안에 있는 td태그를 가져옵니다.
                        tds = list(trs[k].find_elements_by_tag_name('td'))
                        
                        #날짜부터 날씨까지 모든 데이터를 변수에 저장합니다.
                        # temp.append(tds[0].text)
                        day = tds[0].text
                        # temp.append(tds[1].text)
                        temperature = tds[1].text
                        # temp.append(tds[4].text)
                        rain = tds[4].text
                        # temp.append(tds[5].text)
                        snow = tds[5].text
                        # temp.append(tds[6].text)
                        wind = tds[6].text
                        # temp.append(tds[7].text)
                        humidity = tds[7].text
                        # temp.append(tds[8].text)
                        cloud = tds[8].text
                        # temp.append(tds[9].text)
                        sun = tds[9].text
                        # temp.append(tds[10].text)
                        weather = tds[10].text
                        #temp라는 이름의 리스트에 위에서 뽑아낸 값을 넣어줍니다.
                        temp.append([day,temperature,rain,snow,wind,humidity,cloud,sun,weather])
                
                
    #csv에 저장합니다.
    # encoding='cp949' 한글이 깨지는 것을 방지하기 위해 바꿔주었습니다.
    data = pd.DataFrame(temp)
    data.to_csv('인공지능준비.csv', encoding='cp949')

crawling()
