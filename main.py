import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import multiprocessing
import time
from multiprocessing import Pool 
import csv
from concurrent.futures import ThreadPoolExecutor

def get_second_num(raw):
    new = raw.text.split()
    return new[2]

def get_first_num(raw):
    new = raw.text.split()
    return new[0]

def get_decision(detail):
    new = detail.text
    new = new.replace('.','')
    return [s for s in new.split() if s.isdigit()]


def get_links():
    links =[]
    base = "http://ufcstats.com"
    website = f"{base}/statistics/events/completed?page=all"
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
    if(result.status_code != 200):
        print(result.status_code, "Status_code")
        time.sleep(0.5)
        return get_response(url)
    content = result.text
    soup = BeautifulSoup(content, "lxml")
    
    totalStrikesAt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(5) .b-fight-details__table-text:nth-child(1)")
    opponentAt = soup.select_one(".js-fight-section .b-fight-details__table-text:nth-child(1) .b-link_style_black")
    KDA = soup.select('.js-fight-table .l-page_align_left+ .b-fight-details__table-col .b-fight-details__table-text:nth-child(1)')
    TDA = soup.select('.js-fight-table .b-fight-details__table-col:nth-child(6) .b-fight-details__table-text:nth-child(1)')
    SubA =soup.select('.js-fight-table .b-fight-details__table-col:nth-child(8) .b-fight-details__table-text:nth-child(1)')
    CtrlA = soup.select('.js-fight-table .b-fight-details__table-col:nth-child(10) .b-fight-details__table-text:nth-child(1)')
    HeadA = soup.select('table+ .js-fight-section .b-fight-details__table-col:nth-child(4) .b-fight-details__table-text:nth-child(1)')

    totalStrikesBt = soup.select_one(".js-fight-section .b-fight-details__table-col:nth-child(5) .b-fight-details__table-text+ .b-fight-details__table-text")
    opponentBt = soup.select_one(".js-fight-section .b-fight-details__table-text+ .b-fight-details__table-text .b-link_style_black")
    KDB = soup.select('.js-fight-table .l-page_align_left+ .b-fight-details__table-col .b-fight-details__table-text+ .b-fight-details__table-text')
    TDB = soup.select('.js-fight-table .b-fight-details__table-col:nth-child(6) .b-fight-details__table-text+ .b-fight-details__table-text')
    SubB =soup.select('.js-fight-table .b-fight-details__table-col:nth-child(8) .b-fight-details__table-text+ .b-fight-details__table-text')
    CtrlB = soup.select('.js-fight-table .b-fight-details__table-col:nth-child(10) .b-fight-details__table-text+ .b-fight-details__table-text')
    HeadB =soup.select('table+ .js-fight-section .b-fight-details__table-col:nth-child(4) .b-fight-details__table-text+ .b-fight-details__table-text')

    methodt = soup.select_one(".b-fight-details__label+ i")
    opponentAdiv = soup.select_one(".b-fight-details__person:nth-child(1)")
    rounds = soup.select_one(".b-fight-details__text-item:nth-child(4)")
    Arounds = soup.select(".b-fight-details__bar-chart-text_style_light-red")
    Brounds = soup.select(".b-fight-details__bar-chart-text_style_light-blue")
    roundEnd = soup.select_one(".b-fight-details__text-item_first+ .b-fight-details__text-item")
    endRound =int(roundEnd.text.split()[1])
    numrounds = rounds.text.split()[2]

    if(numrounds=='No'):
        dataList.append("Error_skip")
        return dataList
    
    if(((int(numrounds)!=5) & (int(numrounds)!= 3))| (totalStrikesAt is None)):
        dataList.append("Error_skip")
        return dataList
    
    dataList.append(numrounds)

    winLoss = opponentAdiv.find('i')            
    if(winLoss.text.strip()=="W"):
        dataList.append(1)
    else:
        dataList.append(0)

    decision = False
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
        decision =True
    else:
        dataList.append(4)#DQ/NC/Draw

    dataList.append(opponentAt.text.strip())
    dataList.append(get_second_num(totalStrikesAt))
    for i in range(endRound):
        dataList.append(get_second_num(Arounds[i]))
        dataList.append(get_first_num(Arounds[i]))
        dataList.append(KDA[i].text.strip())
        dataList.append(get_first_num(TDA[i]))
        dataList.append(SubA[i].text.strip())
        dataList.append(CtrlA[i].text.strip())
        dataList.append(get_first_num(HeadA[i]))

    if(endRound != 5):
        for i in range((5-endRound)*7):
            dataList.append('--')

    
    dataList.append(opponentBt.text.strip())
    dataList.append(get_second_num(totalStrikesBt))
    for i in range(endRound):
        dataList.append(get_second_num(Brounds[i]))
        dataList.append(get_first_num(Brounds[i]))
        dataList.append(KDB[i].text.strip())
        dataList.append(get_first_num(TDB[i]))
        dataList.append(SubB[i].text.strip())
        dataList.append(CtrlB[i].text.strip())
        dataList.append(get_first_num(HeadB[i]))
    if(endRound != 5):
        for i in range((5-endRound)*7):
            dataList.append('--')
    if(decision):
        details = soup.select(".b-fight-details__text+ .b-fight-details__text .b-fight-details__text-item")
        if(details is not None):
            for detail in details:
                dataList.append(get_decision(detail)[0] +' '+ get_decision(detail)[1])
    
    return dataList
    

def main():
    start_time = time.time()
    cores = multiprocessing.cpu_count()
    linkList = get_links()
    with Pool(int(cores/2)) as p:
        rows= p.map(get_response, linkList)
    with open('UFC.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Number of Rounds','Opponent A Wins','Method',
                         'Opponent A','Total Strikes Attempted (A)',
                         'Round 1 Sig Strikes Attempted (A)','Round 1 Sig Strikes Landed (A)','Round 1 KD (A)','Round 1 TD (A)', 'Round 1 Sub Attempts (A)','Round 1 Ctrl Time (A)', 'Round 1 Head Strikes (A)',
                         'Round 2 Sig Strikes Attempted (A)','Round 2 Sig Strikes Landed (A)','Round 2 KD (A)','Round 2 TD (A)', 'Round 2 Sub Attempts (A)','Round 2 Ctrl Time (A)', 'Round 2 Head Strikes (A)',
                         'Round 3 Sig Strikes Attempted (A)','Round 3 Sig Strikes Landed (A)','Round 3 KD (A)','Round 3 TD (A)', 'Round 3 Sub Attempts (A)','Round 3 Ctrl Time (A)', 'Round 3 Head Strikes (A)',
                         'Round 4 Sig Strikes Attempted (A)','Round 4 Sig Strikes Landed (A)','Round 4 KD (A)','Round 4 TD (A)', 'Round 4 Sub Attempts (A)','Round 4 Ctrl Time (A)', 'Round 4 Head Strikes (A)',
                         'Round 5 Sig Strikes Attempted (A)','Round 5 Sig Strikes Landed (A)','Round 5 KD (A)','Round 5 TD (A)', 'Round 5 Sub Attempts (A)','Round 5 Ctrl Time (A)', 'Round 5 Head Strikes (A)',
                         'Opponent B','Total Strikes Attempted (B)',
                         'Round 1 Sig Strikes Attempted (B)','Round 1 Sig Strikes Landed (B)','Round 1 KD (B)','Round 1 TD (B)', 'Round 1 Sub Attempts (B)','Round 1 Ctrl Time (B)', 'Round 1 Head Strikes (B)',
                         'Round 2 Sig Strikes Attempted (B)','Round 2 Sig Strikes Landed (B)','Round 2 KD (B)','Round 2 TD (B)', 'Round 2 Sub Attempts (B)','Round 2 Ctrl Time (B)', 'Round 2 Head Strikes (B)',
                         'Round 3 Sig Strikes Attempted (B)','Round 3 Sig Strikes Landed (B)','Round 3 KD (B)','Round 3 TD (B)', 'Round 3 Sub Attempts (B)','Round 3 Ctrl Time (B)', 'Round 3 Head Strikes (B)',
                         'Round 4 Sig Strikes Attempted (B)','Round 4 Sig Strikes Landed (B)','Round 4 KD (B)','Round 4 TD (B)', 'Round 4 Sub Attempts (B)','Round 4 Ctrl Time (B)', 'Round 4 Head Strikes (B)',
                         'Round 5 Sig Strikes Attempted (B)','Round 5 Sig Strikes Landed (B)','Round 5 KD (B)','Round 5 TD (B)', 'Round 5 Sub Attempts (B)','Round 5 Ctrl Time (B)', 'Round 5 Head Strikes (B)',
                         'Decision1','Decision2','Decision3', 
                         ])
        for row in rows:
            if(row[0]!='Error_skip'):
                writer.writerow(row)
    print(f"{(time.time() - start_time):.2f} seconds")
    
if __name__ == '__main__':
	main()