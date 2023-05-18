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



def get_second_num(raw):
    new = raw.text.split()
    return new[2]

def get_first_num(raw):
    new = raw.text.split()
    return new[0]

def init_processes(a,b,c,d,e,f,g,h,i, j,k,l,m,n,o,p):
    global Awins
    Awins = a
    global opponentA
    opponentA = b
    global method
    method = c
    global totalStrikesAttA
    totalStrikesAttA = d
    global totalStrikesLandA
    totalStrikesLandA = e
    global TDA
    TDA = f
    global CtrlTimeA
    CtrlTimeA = g
    global significantStrikesA
    significantStrikesA = h
    global KDA
    KDA = i

    global opponentB
    opponentB = j
    global totalStrikesAttB
    totalStrikesAttB = k
    global totalStrikesLandB
    totalStrikesLandB = l
    global TDB
    TDB = m
    global CtrlTimeB
    CtrlTimeB = n
    global significantStrikesB
    significantStrikesB = o
    global KDB
    KDB = p

def get_links():
    links =[]
    base = "http://ufcstats.com"
    website = f"{base}/statistics/events/completed"
    result3 = requests.get(website)
    content3 = result3.text
    soup3 = BeautifulSoup(content3, "html.parser")
    eventBox = soup3.find('table', class_="b-statistics__table-events")

    eventslinks = [link['href'] for link in eventBox.find_all('a', attrs={'class':'b-link b-link_style_black'} )]
    for event in eventslinks:
        result2 = requests.get(event)
        content2 = result2.text
        soup2 = BeautifulSoup(content2, "html.parser")
        box = soup2.find('tbody', class_="b-fight-details__table-body")
        rows = box.find_all('tr', attrs={'class':'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'} )
        for row in rows:
            links.append(row['data-link'])
    return links


def get_response(url):
    result = requests.get(url)
    content = result.text
    soup = BeautifulSoup(content, "html.parser")
    
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
    winLoss = opponentAdiv.find('i')
    if(winLoss.text.strip()=="W"):
        Awins.append(1)
    else:
        Awins.append(0)
    
    totalStrikesAttA.append(get_second_num(totalStrikesAt))
    totalStrikesLandA.append(get_first_num(totalStrikesAt))
    TDA.append(get_first_num(TDAt))
    CtrlTimeA.append(CtrlTimeAt.text.strip())
    significantStrikesA.append(get_first_num(significantStrikesAt))
    KDA.append(KDAt.text.strip())
    opponentA.append(opponentAt.text.strip())
    
    totalStrikesAttB.append(get_second_num(totalStrikesBt))
    totalStrikesLandB.append(get_first_num(totalStrikesBt))
    TDB.append(get_first_num(TDBt))
    CtrlTimeB.append(CtrlTimeBt.text.strip())
    significantStrikesB.append(get_first_num(significantStrikesBt))
    KDB.append(KDBt.text.strip())
    opponentB.append(opponentBt.text.strip())

    methodWords = methodt.text.split()
    if(methodWords[0]== "KO/TKO"):
        method.append(0)
    elif(methodWords[0]== "Submission"):
        method.append(1)
    elif(methodWords[0]=="Decision"):
        if(methodWords[2]=="Unanimous"):
            method.append(2)
        else:
            method.append(3)#split
    else:
        method.append(4)#DQ/NC
    

def main():
    start_time = time.time()
    linkList = get_links()
    cores = multiprocessing.cpu_count()
    manager = Manager()

    Awins = manager.list()
    opponentA = manager.list()
    method = manager.list()
    totalStrikesAttA = manager.list()
    totalStrikesLandA = manager.list()
    TDA = manager.list()
    CtrlTimeA = manager.list()
    significantStrikesA = manager.list()     
    KDA =manager.list()

    opponentB = manager.list()
    totalStrikesAttB = manager.list()
    totalStrikesLandB = manager.list()
    TDB = manager.list()
    CtrlTimeB = manager.list()
    significantStrikesB = manager.list()     
    KDB =manager.list()
    
    with ThreadPoolExecutor(max_workers=15, initializer=init_processes, initargs=(Awins,opponentA,method,totalStrikesAttA,totalStrikesLandA,TDA,CtrlTimeA,significantStrikesA,KDA, opponentB,totalStrikesAttB,totalStrikesLandB,TDB,CtrlTimeB,significantStrikesB,KDB)) as p:
        p.map(get_response, linkList)

    rows = zip(Awins, method, opponentA, totalStrikesAttA, totalStrikesLandA,TDA,CtrlTimeA,significantStrikesA,KDA, opponentB, totalStrikesAttB, totalStrikesLandB,TDB,CtrlTimeB,significantStrikesB,KDB)
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