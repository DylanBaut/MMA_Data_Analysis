from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_second_num(raw):
    new = raw.text.split()
    return new[2]

def get_first_num(raw):
    new = raw.text.split()
    return new[0]

opponentA =[]
opponentB =[]
method = []
Awins = []

totalStrikesAttA = []
totalStrikesLandA = []
TDA = []
CtrlTimeA = []
significantStrikesA = []
KDA =[]

totalStrikesAttB = []
totalStrikesLandB = []
TDB = []
CtrlTimeB = []
significantStrikesB = []
KDB = []

result = requests.get("http://ufcstats.com/fight-details/e3aad51099a23ba4")
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



print("A")
print("/n opponent:")
print(opponentA)
print("/n total strikes attempted:")
print(totalStrikesAttA)
print("/n total strikes landed:")
print(totalStrikesLandA)
print("/n takedowns:")
print(TDA)
print("/n control Time:") 
print(CtrlTimeA)
print("/n significant strikes:")
print(significantStrikesA)
print("/n knockdowns:")
print(KDA)


print("B")
print("/n opponent:")
print(opponentB)
print("/n total strikes attempted:")
print(totalStrikesAttB)
print("/n total strikes landed:")
print(totalStrikesLandB)
print("/n takedowns:")
print(TDB)
print("/n control Time:") 
print(CtrlTimeB)
print("/n significant strikes:")
print(significantStrikesB)
print("/n knockdowns:")
print(KDB)
print("/n method:")
print(method)
print("A is winner")
print(Awins)