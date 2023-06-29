import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import multiprocessing
import time
from multiprocessing import Pool 
import csv

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
    yearslinks =[]
    events = [] 
    base = "http://mmadecisions.com/"
    website = f"{base}decisions-by-event/"
    result3 = requests.get(website)
    content3 = result3.text
    soup3 = BeautifulSoup(content3, "lxml")
    list2 = soup3.find_all('td', class_="list2")
    yearslinks.append('decisions-by-event/')
    
    for listObject in list2:
        temp =listObject.find('a')
        if(int(temp.text)>2008):
            yearslinks.append(temp.get('href'))

    for year in yearslinks:
        website = f"{base}{year}"
        result2 = requests.get(website)
        content2 = result2.text
        soup2 = BeautifulSoup(content2, "lxml")
        eventsBlocks = soup2.find_all('tr', class_='decision')
        for eventsBlock in eventsBlocks:
            title = eventsBlock.find('a')
            if(title.text.find('UFC') != -1):
                events.append(title.get('href'))
    for event in events:
        website = f"{base}{event}"
        result = requests.get(website)
        content = result.text
        soup = BeautifulSoup(content, "lxml")
        fights = soup.find_all('td', class_='list2')
        for fight in fights:
            temp = fight.find('a')
            links.append(base+temp.get('href'))
    return links


def get_response(url):
    dataList =[]
    result = requests.get(url.strip())
    
    if(result.status_code != 200):
        print(result.status_code, "Status_code")
        time.sleep(0.5)
        return get_response(url)

    content = result.text
    soup = BeautifulSoup(content, "lxml")
    tied =soup.select_one('.decision-middle i')
    tied = tied.text.strip()

    winner =soup.select_one('.decision-top a')
    winner = winner.text.strip()
    opponentA =winner.replace(u'\xa0', ' ')
    winner =winner[(winner.find(u'\xa0')+1):]
    if(tied =='defeats'):
        dataList.append(winner) #winner or first name = A
    else:
        dataList.append('DRAW')
        winner='DRAW'
    
    opponB =soup.select_one('.decision-bottom a')
    opponentB = opponB.text.strip()
    opponentB =opponentB.replace(u'\xa0', ' ')

    dataList.append(opponentA)
    dataList.append(opponentB)
    judges = soup.find_all('td', {'width':'33%'})
    scrap = False
    for x in range(3):
        scores = judges[x].find_all('tr', {'class':'decision'})
        rounds=len(scores)
        for y in range(5):
            if(y<rounds):
                vals = scores[y].text.split()
                if(vals[1]=='-'):
                    scrap =True
                    dataList.append('--')
                elif(vals[1]=='10'):
                    if(vals[2]=='9'):      # A is winner, B is loser
                        dataList.append(1) #1= A gets 10-9
                    elif(vals[2]=='8'):
                        dataList.append(2) #2= A gets 10-8
                    elif(vals[2]=='7'):
                        dataList.append(3)#3= A gets 10-7
                    elif(vals[2]=='10'):
                        dataList.append(0)#0= A gets 10-10
                else:
                    if(vals[1]=='9'):
                        dataList.append(-1) #1= A gets 10-9
                    elif(vals[1]=='8'):
                        dataList.append(-2) #2= A gets 10-8
                    elif(vals[1]=='7'):
                        dataList.append(-3)#3= A gets 10-7
                    
            else:
                dataList.append('--')
    if(scrap):
        dataList.append('Error_skip')

    table = soup.find('table', {'style':'border-spacing: 0px; width: 100%'})
    if(table is not None):
        medias = table.find_all('tr',{'class': 'decision'})
        if(medias is not None):
            if(len(medias)>=5):
                agree =0
                tot =0
                for media in medias:
                    words = media.text.strip().split('\n')
                    if (words[len(words)-1] == winner):
                        agree+=1
                    tot+=1
                dataList.append(round((agree/tot),3))
            else:
                dataList.append('--')
        else:
            dataList.append('--')
    else:
        dataList.append('--')
        
    scoresFull = soup.select('td td td .bottom-cell b')
    for idx in range(len(scoresFull)):
        scoresFull[idx]=scoresFull[idx].text.strip()
    dataList.append(scoresFull)

    scorecards =soup.select_one('#scorecards_submitted td')
    if (int(get_first_num(scorecards.select_one('b'))) >= 30):
        fansection = soup.find('td', {'valign':'top','colspan':'6'})
        fanrounds =fansection.find_all('td', {'width':'33%','valign':'top'})
        for fanround in fanrounds:
            fanscores =fanround.find_all('tr', {'class':'decision'})
            roundsum=0
            for fanscore in fanscores:
                words = fanscore.text.strip().split('\n')
                percentage = words[2].strip()
                decimal = round((float(percentage[0: len(percentage)-1])/100),3)
                if(words[1].strip()== winner):
                    if(words[0].strip() == '10-9'):
                        roundsum += 1* decimal
                    elif(words[0].strip() == '10-8'):
                        roundsum += 2* decimal
                    elif(words[0].strip() == '10-7'):
                        roundsum += 3* decimal
                else:
                    if(words[0].strip() == '10-9'):
                        roundsum -= 1* decimal
                    elif(words[0].strip() == '10-8'):
                        roundsum -= 2* decimal
                    elif(words[0].strip() == '10-7'):
                        roundsum -= 3* decimal
            dataList.append(round(roundsum, 3))
    return dataList

def main():
    start_time = time.time()    
    cores = multiprocessing.cpu_count()
    linkList = get_links()

    with Pool(int(cores/2)) as p:
        rows= p.map(get_response, linkList)

    with open('decisions.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Winner', 'Opponent_A', 'Opponent_B', 'Rd1A', 'Rd2A', 'Rd3A', 'Rd4A', 'Rd5A', 'Rd1B', 'Rd2B', 'Rd3B', 'Rd4B', 'Rd5B', 'Rd1C', 'Rd2C', 'Rd3C', 'Rd4C', 
                         'Rd5C', 'Media_score_ratio_of_agreement','Full_Scores', 'Rd1_Fans', 'Rd2_Fans', 'Rd3_Fans', 'Rd4_Fans', 'Rd5_Fans'
                         ])
        for row in rows:
            if(row[18]!='Error_skip'):
                writer.writerow(row)
    print(f"{(time.time() - start_time):.2f} seconds")

if __name__ == '__main__':
	main()