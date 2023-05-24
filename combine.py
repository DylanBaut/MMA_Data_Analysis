import pandas as pd
import csv
from unidecode import unidecode
from difflib import SequenceMatcher
import time


def main():
    start_time = time.time()
    stats = pd.read_csv('UFC.csv')
    decisions = pd.read_csv('decisions.csv') 
    index=0
    lost=0
    decisionsSpot =0
    lostlist =[]
    idxmax = max(stats.count())
    with open('output.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Number_of_Rounds','Opponent_A_Wins','Method',
                        'Opponent_A','Total_Strikes_Attempted_(A)',
                        'Round_1_Sig_Strikes_Attempted_(A)','Round_1_Sig_Strikes_Landed_(A)','Round_1_KD_(A)','Round_1_TD_(A)','Round_1_Sub_Attempts_(A)','Round_1_Ctrl_Time_(A)','Round_1_Head_Strikes_(A)',
                        'Round_2_Sig_Strikes_Attempted_(A)','Round_2_Sig_Strikes_Landed_(A)','Round_2_KD_(A)','Round_2_TD_(A)','Round_2_Sub_Attempts_(A)','Round_2_Ctrl_Time_(A)','Round_2_Head_Strikes_(A)',
                        'Round_3_Sig_Strikes_Attempted_(A)','Round_3_Sig_Strikes_Landed_(A)','Round_3_KD_(A)','Round_3_TD_(A)','Round_3_Sub_Attempts_(A)','Round_3_Ctrl_Time_(A)','Round_3_Head_Strikes_(A)',
                        'Round_4_Sig_Strikes_Attempted_(A)','Round_4_Sig_Strikes_Landed_(A)','Round_4_KD_(A)','Round_4_TD_(A)','Round_4_Sub_Attempts_(A)','Round_4_Ctrl_Time_(A)','Round_4_Head_Strikes_(A)',
                        'Round_5_Sig_Strikes_Attempted_(A)','Round_5_Sig_Strikes_Landed_(A)','Round_5_KD_(A)','Round_5_TD_(A)','Round_5_Sub_Attempts_(A)','Round_5_Ctrl_Time_(A)','Round_5_Head_Strikes_(A)',
                        'Opponent_B','Total_Strikes_Attempted_(B)',
                        'Round_1_Sig_Strikes_Attempted_(B)','Round_1_Sig_Strikes_Landed_(B)','Round_1_KD_(B)','Round_1_TD_(B)','Round_1_Sub_Attempts_(B)','Round_1_Ctrl_Time_(B)','Round_1_Head_Strikes_(B)',
                        'Round_2_Sig_Strikes_Attempted_(B)','Round_2_Sig_Strikes_Landed_(B)','Round_2_KD_(B)','Round_2_TD_(B)','Round_2_Sub_Attempts_(B)','Round_2_Ctrl_Time_(B)','Round_2_Head_Strikes_(B)',
                        'Round_3_Sig_Strikes_Attempted_(B)','Round_3_Sig_Strikes_Landed_(B)','Round_3_KD_(B)','Round_3_TD_(B)','Round_3_Sub_Attempts_(B)','Round_3_Ctrl_Time_(B)','Round_3_Head_Strikes_(B)',
                        'Round_4_Sig_Strikes_Attempted_(B)','Round_4_Sig_Strikes_Landed_(B)','Round_4_KD_(B)','Round_4_TD_(B)','Round_4_Sub_Attempts_(B)','Round_4_Ctrl_Time_(B)','Round_4_Head_Strikes_(B)',
                        'Round_5_Sig_Strikes_Attempted_(B)','Round_5_Sig_Strikes_Landed_(B)','Round_5_KD_(B)','Round_5_TD_(B)','Round_5_Sub_Attempts_(B)','Round_5_Ctrl_Time_(B)','Round_5_Head_Strikes_(B)',
                        'Decision1','Decision2','Decision3','Winner', 'Opponent A', 'Opponent B', 'Rd1A', 'Rd2A', 'Rd3A', 'Rd4A', 'Rd5A', 'Rd1B', 'Rd2B', 'Rd3B', 'Rd4B', 'Rd5B', 'Rd1C', 'Rd2C', 'Rd3C', 'Rd4C', 
                         'Rd5C', 'Media score ratio of agreement', 'Rd1 Fans', 'Rd2 Fans', 'Rd3 Fans', 'Rd4 Fans', 'Rd5 Fans'
                         ])
        for row in decisions.itertuples(index =True):
            skipped = False
            idx=index
            while(not((( SequenceMatcher(a= (unidecode((getattr(row, 'Opponent_A')))).lower(),b=(unidecode(stats.loc[idx, :].values.flatten().tolist()[3])).lower()).ratio()  >= 0.8 )&( SequenceMatcher(a= (unidecode((getattr(row, 'Opponent_B')))).lower(),b=(unidecode(stats.loc[idx, :].values.flatten().tolist()[40])).lower()).ratio()  >= 0.8 ))|
                       (( SequenceMatcher(a= (unidecode((getattr(row, 'Opponent_B')))).lower(),b=(unidecode(stats.loc[idx, :].values.flatten().tolist()[3])).lower()).ratio()  >= 0.8 )&( SequenceMatcher(a= (unidecode((getattr(row, 'Opponent_A')))).lower(),b=(unidecode(stats.loc[idx, :].values.flatten().tolist()[40])).lower()).ratio()  >= 0.8 ))) ):
                if(idx+1 >=idxmax):
                    skipped = True
                    lost+=1
                    lostlist.append(decisionsSpot)
                    break
                else:
                    idx+=1
            decisionsSpot +=1
            if(not skipped):
                temp = stats.loc[idx, :].values.flatten().tolist()
                decs= list(row)
                temp.extend(decs[1:2])
                temp.extend(decs[4:])
                writer.writerow(temp)
                index=idx
        print(f"{lost} unmatched bouts lost when merging at indices {lostlist}")
        print(f"{(time.time() - start_time):.2f} seconds")

main()
