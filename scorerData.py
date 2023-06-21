import pandas as pd
import re
from unidecode import unidecode
from difflib import SequenceMatcher
output_df = pd.read_csv('output.csv')
scorer_columns= ['Opponent A','Opponent B',
                        'Round_1_Sig_Strikes_Attempted_(A)','Round_1_Sig_Strikes_Landed_(A)','Round_1_KD_(A)','Round_1_TD_(A)','Round_1_Sub_Attempts_(A)','Round_1_Ctrl_Time_(A)','Round_1_Head_Strikes_(A)',
                        'Round_2_Sig_Strikes_Attempted_(A)','Round_2_Sig_Strikes_Landed_(A)','Round_2_KD_(A)','Round_2_TD_(A)','Round_2_Sub_Attempts_(A)','Round_2_Ctrl_Time_(A)','Round_2_Head_Strikes_(A)',
                        'Round_3_Sig_Strikes_Attempted_(A)','Round_3_Sig_Strikes_Landed_(A)','Round_3_KD_(A)','Round_3_TD_(A)','Round_3_Sub_Attempts_(A)','Round_3_Ctrl_Time_(A)','Round_3_Head_Strikes_(A)',
                        'Round_4_Sig_Strikes_Attempted_(A)','Round_4_Sig_Strikes_Landed_(A)','Round_4_KD_(A)','Round_4_TD_(A)','Round_4_Sub_Attempts_(A)','Round_4_Ctrl_Time_(A)','Round_4_Head_Strikes_(A)',
                        'Round_5_Sig_Strikes_Attempted_(A)','Round_5_Sig_Strikes_Landed_(A)','Round_5_KD_(A)','Round_5_TD_(A)','Round_5_Sub_Attempts_(A)','Round_5_Ctrl_Time_(A)','Round_5_Head_Strikes_(A)',
                        'Round_1_Sig_Strikes_Attempted_(B)','Round_1_Sig_Strikes_Landed_(B)','Round_1_KD_(B)','Round_1_TD_(B)','Round_1_Sub_Attempts_(B)','Round_1_Ctrl_Time_(B)','Round_1_Head_Strikes_(B)',
                        'Round_2_Sig_Strikes_Attempted_(B)','Round_2_Sig_Strikes_Landed_(B)','Round_2_KD_(B)','Round_2_TD_(B)','Round_2_Sub_Attempts_(B)','Round_2_Ctrl_Time_(B)','Round_2_Head_Strikes_(B)',
                        'Round_3_Sig_Strikes_Attempted_(B)','Round_3_Sig_Strikes_Landed_(B)','Round_3_KD_(B)','Round_3_TD_(B)','Round_3_Sub_Attempts_(B)','Round_3_Ctrl_Time_(B)','Round_3_Head_Strikes_(B)',
                        'Round_4_Sig_Strikes_Attempted_(B)','Round_4_Sig_Strikes_Landed_(B)','Round_4_KD_(B)','Round_4_TD_(B)','Round_4_Sub_Attempts_(B)','Round_4_Ctrl_Time_(B)','Round_4_Head_Strikes_(B)',
                        'Round_5_Sig_Strikes_Attempted_(B)','Round_5_Sig_Strikes_Landed_(B)','Round_5_KD_(B)','Round_5_TD_(B)','Round_5_Sub_Attempts_(B)','Round_5_Ctrl_Time_(B)','Round_5_Head_Strikes_(B)',
                 'Decision1','Decision1B', 'Decision2','Decision2B','Decision3','Decision3B',
                 ]
scorer_df = pd.DataFrame(columns=scorer_columns)
for rowtuple in output_df.itertuples():
    row= list(rowtuple)[1:]
    rowList =[]
    Awins =row[1]
    oppA = row[3]
    oppB = row[40]
    roundinfoA = row[5:40]
    roundinfoB = row[42:77]
    decisions =row[98]
    decisionsOppA = row[81]
    rowList.append(oppA)  
    rowList.append(oppB)
    rowList.extend(roundinfoA)
    rowList.extend(roundinfoB)
    if(row[80]=='DRAW'):
        # if Opponent A of UFC is Opponent A in decisions
        nums = re.findall(r'\d+',decisions)
        if(SequenceMatcher(a=unidecode(row[81]).lower(), b=unidecode(row[3]).lower()).ratio() >=0.8):
            rowList.append(nums[0])
            rowList.append(nums[1])
            rowList.append(nums[2])
            rowList.append(nums[3])
            rowList.append(nums[4])
            rowList.append(nums[5])
        else:
            rowList.append(nums[1])
            rowList.append(nums[0])
            rowList.append(nums[3])
            rowList.append(nums[2])
            rowList.append(nums[5])
            rowList.append(nums[4])
    else:
        nums = re.findall(r'\d+',decisions)
        if(int(Awins)==1):
            rowList.append(nums[0])
            rowList.append(nums[1])
            rowList.append(nums[2])
            rowList.append(nums[3])
            rowList.append(nums[4])
            rowList.append(nums[5])
        else:
            rowList.append(nums[1])
            rowList.append(nums[0])
            rowList.append(nums[3])
            rowList.append(nums[2])
            rowList.append(nums[5])
            rowList.append(nums[4])
    scorer_df.loc[len(scorer_df.index)] = rowList
scorer_df.to_csv( 'scoresData.csv', index=False)

#A vs B:
# 49- 46: A -B
