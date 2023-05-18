from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import time
from multiprocessing import Pool, Manager 
import csv
from concurrent.futures import ThreadPoolExecutor
from inspect import getmembers
import random

def get_second_num(raw):
    new = raw.text.split()
    return new[2]

def get_first_num(raw):
    new = raw.text.split()
    return new[0]


def get_links():
    links =[]
    base = "http://ufcstats.com"
    website = f"{base}/statistics/events/completed"
    result3 = requests.get(website)
    content3 = result3.text
    soup3 = BeautifulSoup(content3, "lxml")
    eventBox = soup3.find('table', class_="b-statistics__table-events")

    eventslinks = [link['href'] for link in eventBox.find_all('a', attrs={'class':'b-link b-link_style_black'} )]
    for event in eventslinks:
        result2 = requests.get(event)
        content2 = result2.text
        soup2 = BeautifulSoup(content2, "lxml")
        box = soup2.find('tbody', class_="b-fight-details__table-body")
        rows = box.find_all('tr', attrs={'class':'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'} )
        for row in rows:
            links.append(row['data-link'])
    return links


def get_response(url):
    dataList =[]

    result = requests.get(url)
    if(result.status_code==429):
        time.sleep(0.1)
        return get_response(url)
    content = result.text
    soup = BeautifulSoup(content, "lxml")
    
    totalStrikesAt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(5) .b-fight-details__table-text:nth-child(1)")
    TDAt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(6) .b-fight-details__table-text:nth-child(1)")
    CtrlTimeAt = soup.select_one(".b-fight-details__table-col:nth-child(10) .b-fight-details__table-text:nth-child(1)")
    significantStrikesAt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(3) .b-fight-details__table-text:nth-child(1)")
    KDAt = soup.select_one(".js-fight-section .l-page_align_left+ .b-fight-details__table-col .b-fight-details__table-text:nth-child(1)")
    opponentAt = soup.select_one(".js-fight-section .b-fight-details__table-text:nth-child(1) .b-link_style_black")
    
    totalStrikesBt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(5) .b-fight-details__table-text+ .b-fight-details__table-text")
    TDBt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(6) .b-fight-details__table-text+ .b-fight-details__table-text")
    CtrlTimeBt = soup.select_one(".b-fight-details__table-col:nth-child(10) .b-fight-details__table-text+ .b-fight-details__table-text")
    significantStrikesBt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(3) .b-fight-details__table-text+ .b-fight-details__table-text")
    KDBt = soup.select_one(".js-fight-section .l-page_align_left+ .b-fight-details__table-col .b-fight-details__table-text+ .b-fight-details__table-text")
    opponentBt = soup.select_one(".js-fight-section .b-fight-details__table-text+ .b-fight-details__table-text .b-link_style_black")
    methodt = soup.select_one(".b-fight-details__label+ i")
    opponentAdiv = soup.select_one(".b-fight-details__person:nth-child(1)")
    #if(opponentAdiv is None):
    #    print(getmembers(result))
    #    print(result.status_code)

    winLoss = opponentAdiv.find('i')            
    if(winLoss.text.strip()=="W"):
        dataList.append(1)
    else:
        dataList.append(0)
    
    methodWords = methodt.text.split()
    if(methodWords[0]== "KO/TKO"):
        dataList.append(0)
    elif(methodWords[0]== "Submission"):
        dataList.append(1)
    elif(methodWords[0]=="Decision"):
        if(methodWords[2]=="Unanimous"):
            dataList.append(2)
        else:
            dataList.append(3)#split
    else:
        dataList.append(4)#DQ/NC

    dataList.append(opponentAt.text.strip())
    dataList.append(get_second_num(totalStrikesAt))
    dataList.append(get_first_num(totalStrikesAt))
    dataList.append(get_first_num(TDAt))
    dataList.append(CtrlTimeAt.text.strip())
    dataList.append(get_first_num(significantStrikesAt))
    dataList.append(KDAt.text.strip())
    
    dataList.append(opponentBt.text.strip())
    dataList.append(get_second_num(totalStrikesBt))
    dataList.append(get_first_num(totalStrikesBt))
    dataList.append(get_first_num(TDBt))
    dataList.append(CtrlTimeBt.text.strip())
    dataList.append(get_first_num(significantStrikesBt))
    dataList.append(KDBt.text.strip())
    return dataList
    

def main():
    start_time = time.time()
    linkList = get_links()
    
    with ThreadPoolExecutor(max_workers=15) as p:
        rows= p.map(get_response, linkList)

    with open('UFC.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Opponent A Wins','method','Opponent A','Total Strikes Attempted (A)','Total Strikes Landed (A)', 'Takedowns (A)', 
                         'Control Time (A)', 'Significant Strikes: (A)','Knockdowns (A)','Opponent B','Total Strikes Attempted (B)','Total Strikes Landed (B)', 'Takedowns (B)', 
                         'Control Time (B)', 'Significant Strikes: (B)','Knockdowns (B)'])
        for row in rows:
            writer.writerow(row)
    print(f"{(time.time() - start_time):.2f} seconds")
    
if __name__ == '__main__':
	main()