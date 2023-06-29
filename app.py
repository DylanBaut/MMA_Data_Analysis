from flask import Flask, render_template, request, flash, redirect, url_for, session
from torch import load, no_grad, tensor, float32
from torch.nn import Sequential, Linear, ReLU
from pandas import read_csv

app = Flask(__name__)
app.secret_key="jaxb"
formData ={}
scores_df = read_csv('static/data/scoresData.csv')

@app.route("/", methods =['POST', 'GET'])
def index():
    flash("Enter Round", "label")
    flash("Information", "label")
    
    if request.method =='POST':
        formData['sigAtt1']=request.form['sigAtt1']
        formData['sigLand1']=request.form['sigLand1']
        formData['KD1']=request.form['KD1']
        formData['TD1']=request.form['TD1']
        formData['sub1']=request.form['sub1']
        formData['ctrl1']=request.form['ctrl1']
        formData['head1']=request.form['head1']
        formData['sigAtt2']=request.form['sigAtt2']
        formData['sigLand2']=request.form['sigLand2']
        formData['KD2']=request.form['KD2']
        formData['TD2']=request.form['TD2']
        formData['sub2']=request.form['sub2']
        formData['ctrl2']=request.form['ctrl2']
        formData['head2']=request.form['head2']
        formData['modelSel']=request.form['modelSel']
        return redirect(url_for('output'))
    else:
        return render_template("index.html")

def get_prediction():
    model1 = Sequential(
    Linear(7, 15),
        ReLU(),
        Linear(15, 15),
        ReLU(),
        Linear(15, 5),
        ReLU(),
        Linear(5, 5),
    )
    model1.load_state_dict(load('static/models/model1.pth'))
    model1.eval()
    
    model2 = Sequential(
        Linear(7, 15),
        ReLU(),
        Linear(15, 15),
        ReLU(),
        Linear(15, 5),
        ReLU(),
        Linear(5, 5),
    )
    model2.load_state_dict(load('static/models/model2.pth'))
    model2.eval()
    inputData =[]
    inputData.append(int(formData['sigAtt1']) - int(formData['sigAtt2']))
    inputData.append(int(formData['sigLand1']) - int(formData['sigLand2']))
    inputData.append(int(formData['KD1']) - int(formData['KD2']))
    inputData.append(int(formData['TD1']) - int(formData['TD2']))
    inputData.append(int(formData['sub1']) - int(formData['sub2']))
    inputData.append(int(formData['ctrl1']) - int(formData['ctrl2']))
    inputData.append(int(formData['head1']) - int(formData['head2']))
    inputData = tensor(inputData, dtype= float32)
    if(formData['modelSel']=="Judge"):
        with no_grad():
            y_pred =model1(inputData)
            return y_pred
    else:
        with no_grad():
            y_pred =model2(inputData)
            return y_pred
    

@app.route("/output", methods =['POST', 'GET'])
def output():
    result =get_prediction().tolist()
    indexMax = result.index(max(result))
    indexMin = result.index(min(result))
    match indexMax:
        case 0:
            score = '10-10'
            winner = 'both'
        case 1:
            score = '10-9'
            winner = 'Opponent 1'
        case 2:
            score = '10-8'
            winner = 'Opponent 1'
        case 3:
            score = '10-9'
            winner = 'Opponent 2'
        case 4:
            score = '10-8'
            winner = 'Opponent 2'
    session.pop('_flashes', None)
    flash(f"{score} in favor of {winner}", 'label')
    minVal = result[indexMin]
    for idx in range(len(result)):
        result[idx]= result[idx]-minVal
    sumVal = sum(result)
    for idx in range(len(result)):
        result[idx]= round(((result[idx]/sumVal)*100),2)
    flash(result[0],'10-10')
    flash(result[1],'10-9')
    flash(result[2],'10-8')
    flash(result[3],'9-10')
    flash(result[4],'8-10')
    if request.method =='POST':
        formData['sigAtt1']=request.form['sigAtt1']
        formData['sigLand1']=request.form['sigLand1']
        formData['KD1']=request.form['KD1']
        formData['TD1']=request.form['TD1']
        formData['sub1']=request.form['sub1']
        formData['ctrl1']=request.form['ctrl1']
        formData['head1']=request.form['head1']
        formData['sigAtt2']=request.form['sigAtt2']
        formData['sigLand2']=request.form['sigLand2']
        formData['KD2']=request.form['KD2']
        formData['TD2']=request.form['TD2']
        formData['sub2']=request.form['sub2']
        formData['ctrl2']=request.form['ctrl2']
        formData['head2']=request.form['head2']
        formData['modelSel']=request.form['modelSel']
        return redirect(url_for('output'))
    else:
        return render_template("index.html")
    

@app.route('/fightScorer', methods =['POST', 'GET'])
def fightScorer():
    flash(len(scores_df), "len")
    if request.method =='POST':
        formData['fightselect']=request.form['fight-list']
        formData['modelSel']=request.form['modelSel']
        return redirect(url_for('searchOutput'))
    else:
        return render_template("search.html", len =len(scores_df))
    
def get_sec(time_str):
    m, s = time_str.split(':')
    return int(m) * 60 + int(s)

@app.route("/searchOutput", methods =['POST', 'GET'])
def searchOutput():
    row=scores_df.iloc[0]
    if('fightselect' in formData and 'modelSel' in formData):
        row = scores_df.iloc[int(formData['fightselect'])]
        session.pop('_flashes', None)
    else:
        session.pop('_flashes', None)
        flash("CHOICE WAS NOT RECIEVED, TRY CHOOSING AGAIN.", "opponents")
        formData['modelSel']="Judge"
    if(row[24] =="--"):
        roundNum=3
    else:
        roundNum=5
    results =[]
    idxA, idxB, idxC, idxD, idxE, idxF, idxG, idxH, idxI, idxJ, idxK, idxL, idxM, idxN = 2,3,4,5,6,7,8,37,38, 39, 40, 41, 42, 43
    for roundNumber in range(roundNum): 
        formData['sigAtt1']=row[idxA]
        formData['sigLand1']=row[idxB]
        formData['KD1']=row[idxC]
        formData['TD1']=row[idxD]
        formData['sub1']=row[idxE]
        formData['ctrl1']=get_sec(row[idxF])
        formData['head1']=row[idxG]
        formData['sigAtt2']=row[idxH]
        formData['sigLand2']=row[idxI]
        formData['KD2']=row[idxJ]
        formData['TD2']=row[idxK]
        formData['sub2']=row[idxL]
        formData['ctrl2']=get_sec(row[idxM])
        formData['head2']=row[idxN]
        result =get_prediction().tolist()
        results.append(result)
        idxA, idxB, idxC, idxD, idxE, idxF, idxG, idxH, idxI, idxJ, idxK, idxL, idxM, idxN = idxA+7, idxB+7, idxC+7, idxD+7, idxE+7, idxF+7, idxG+7, idxH+7, idxI+7, idxJ+7, idxK+7, idxL+7, idxM+7, idxN+7
    scoreCard =[0,0]
    percentages = []
    for tier in range(len(results)):
        result =results[tier]
        indexMax = result.index(max(result))
        match indexMax:
                case 0:
                    scoreCard[0] = scoreCard[0]+10
                    scoreCard[1] = scoreCard[1]+10
                case 1:
                    scoreCard[0] = scoreCard[0]+10
                    scoreCard[1] = scoreCard[1]+9
                case 2:
                    scoreCard[0] = scoreCard[0]+10
                    scoreCard[1] = scoreCard[1]+8
                case 3:
                    scoreCard[0] = scoreCard[0]+9
                    scoreCard[1] = scoreCard[1]+10
                case 4:
                    scoreCard[0] = scoreCard[0]+8
                    scoreCard[1] = scoreCard[1]+10
        indexMin = result.index(min(result))
        minVal = result[indexMin]
        for idx in range(len(result)):
            result[idx]= result[idx]-minVal
        sumVal = sum(result)
        percentagesDict = {}
        for idx in range(len(result)):
            result[idx]= round(((result[idx]/sumVal)*100),2)
            match idx:
                case 0:
                    scoreVal = '10-10'
                    percentagesDict["0"]= result[idx]
                case 1:
                    scoreVal = '10-9 OppA'
                    percentagesDict["1"]= result[idx]
                case 2:
                    scoreVal = '10-8 OppA'
                    percentagesDict["2"]= result[idx]
                case 3:
                    scoreVal = '10-9 OppB'
                    percentagesDict["3"]= result[idx]
                case 4:
                    scoreVal = '10-8 OppB'
                    percentagesDict["4"]= result[idx]

            flash(scoreVal+": "+str(result[idx])+"%", tier)
        percentages.append(percentagesDict)
    tempList=[]
    for percentage in percentages:
        tempList.append(dict(sorted(percentage.items(), key=lambda x:x[1])[2:]))
    cards = {}
    rd1=tempList[0]
    rd2=tempList[1]
    rd3=tempList[2]
    if(roundNum==5):
        rd4=tempList[3]
        rd5=tempList[4]
    for idx1 in rd1:
        oppScores = [0,0]
        prob = 1
        prob =prob*(float(rd1[idx1])/100)
        match int(idx1):
            case 0:
                oppScores[0] = oppScores[0]+10
                oppScores[1] = oppScores[1]+10
            case 1:
                oppScores[0] = oppScores[0]+10
                oppScores[1] = oppScores[1]+9
            case 2:
                oppScores[0] = oppScores[0]+10
                oppScores[1] = oppScores[1]+8
            case 3:
                oppScores[0] = oppScores[0]+9
                oppScores[1] = oppScores[1]+10
            case 4:
                oppScores[0] = oppScores[0]+8
                oppScores[1] = oppScores[1]+10
        for idx2 in rd2:
            oppScores2 = oppScores.copy()
            prob2 = prob
            prob2 =prob2*(float(rd2[idx2])/100)
            match int(idx2):
                case 0:
                    oppScores2[0] = oppScores2[0]+10
                    oppScores2[1] = oppScores2[1]+10
                case 1:
                    oppScores2[0] = oppScores2[0]+10
                    oppScores2[1] = oppScores2[1]+9
                case 2:
                    oppScores2[0] = oppScores2[0]+10
                    oppScores2[1] = oppScores2[1]+8
                case 3:
                    oppScores2[0] = oppScores2[0]+9
                    oppScores2[1] = oppScores2[1]+10
                case 4:
                    oppScores2[0] = oppScores2[0]+8
                    oppScores2[1] = oppScores2[1]+10
            for idx3 in rd3:
                oppScores3 = oppScores2.copy()
                prob3 = prob2
                prob3 =prob3*(float(rd3[idx3])/100)
                match int(idx3):
                    case 0:
                        oppScores3[0] = oppScores3[0]+10
                        oppScores3[1] = oppScores3[1]+10
                    case 1:
                        oppScores3[0] = oppScores3[0]+10
                        oppScores3[1] = oppScores3[1]+9
                    case 2:
                        oppScores3[0] = oppScores3[0]+10
                        oppScores3[1] = oppScores3[1]+8
                    case 3:
                        oppScores3[0] = oppScores3[0]+9
                        oppScores3[1] = oppScores3[1]+10
                    case 4:
                        oppScores3[0] = oppScores3[0]+8
                        oppScores3[1] = oppScores3[1]+10
                if(roundNum ==3):
                    key = str(str(oppScores3[0])+"-"+str(oppScores3[1]))
                    if(key in cards):
                        if( prob3 > float(cards[key])):
                            cards[key] = prob3
                    else:
                        cards[key] = prob3
                else:
                    for idx4 in rd4:
                        oppScores4 = oppScores3.copy()
                        prob4 = prob3
                        prob4 =prob4*(float(rd4[idx4])/100)
                        match int(idx4):
                            case 0:
                                oppScores4[0] = oppScores4[0]+10
                                oppScores4[1] = oppScores4[1]+10
                            case 1:
                                oppScores4[0] = oppScores4[0]+10
                                oppScores4[1] = oppScores4[1]+9
                            case 2:
                                oppScores4[0] = oppScores4[0]+10
                                oppScores4[1] = oppScores4[1]+8
                            case 3:
                                oppScores4[0] = oppScores4[0]+9
                                oppScores4[1] = oppScores4[1]+10
                            case 4:
                                oppScores4[0] = oppScores4[0]+8
                                oppScores4[1] = oppScores4[1]+10
                        for idx5 in rd5:
                            oppScores5 = oppScores4.copy()
                            prob5 = prob4
                            prob5 =prob5*(float(rd5[idx5])/100)
                            match int(idx5):
                                case 0:
                                    oppScores5[0] = oppScores5[0]+10
                                    oppScores5[1] = oppScores5[1]+10
                                case 1:
                                    oppScores5[0] = oppScores5[0]+10
                                    oppScores5[1] = oppScores5[1]+9
                                case 2:
                                    oppScores5[0] = oppScores5[0]+10
                                    oppScores5[1] = oppScores5[1]+8
                                case 3:
                                    oppScores5[0] = oppScores5[0]+9
                                    oppScores5[1] = oppScores5[1]+10
                                case 4:
                                    oppScores5[0] = oppScores5[0]+8
                                    oppScores5[1] = oppScores5[1]+10
                            key = str(str(oppScores5[0])+"-"+str(oppScores5[1]))
                            if(key in cards):
                                if( prob5 > float(cards[key])):
                                    cards[key] = prob5
                            else:
                                cards[key] = prob5
    cards = dict(sorted(cards.items(), key=lambda x:x[1])[-5:])
    labels =["fifth","fourth","third","second","first"]
    labelIdx=0
    total=0
    for key in cards:
        flash(key +" "+ str(round((cards[key]*100),3))+"%",labels[labelIdx])
        total+=cards[key]
        labelIdx+=1
    for key in cards:
        cards[key] =cards[key]/total
    decs = [str(row[72])+"-"+str(row[73]), str(row[74])+"-"+str(row[75]), str(row[76])+"-"+str(row[77])]
    mean =0
    keyScores = list(cards.keys())
    flash("Opponent A ("+row[0]+") vs. Opponent B ("+row[1]+")", "opponents")
    tiebraker =0
    ties =0
    for dec in decs:
        nums =dec.split("-")
        if(nums[0]>nums[1]):
            tiebraker-=1
        elif(nums[1]>nums[0]):
            tiebraker+=1
        else:
            ties +=1
        flash(dec,"judging")
        if(dec in cards):
            foundidx = keyScores.index(dec)
            match foundidx:
                case 0:
                    mean += cards[dec]*0.56
                case 1:
                    mean += cards[dec]*0.66
                case 2:
                    mean += cards[dec]
                case 3:
                    mean += cards[dec]*1.1
                case 4:
                    mean += cards[dec]*1.2
        else:
            mean += (cards[keyScores[0]]/4)
            print("no match")
    if(ties >=2):
        flash("Tie", "winning")
    elif(tiebraker>0):
        flash(row[1], "winning")
    elif(tiebraker<0):
        flash(row[0], "winning")
    else:
        flash("Tie", "winning")
    mean=mean/3
    print(mean)
    accurLevel = mean *3.65
    flash(accurLevel, "accuracy")
    if(scoreCard[0]>scoreCard[1]):
        winner =row[0]
    else:
        winner =row[1]
    flash(str(scoreCard[0])+" - "+str(scoreCard[1]) +" "+winner, "path")
    if request.method =='POST':
        formData['fightselect']=request.form['fight-list']
        return redirect(url_for('searchOutput'))
    else:
        return render_template("search2.html",len =len(scores_df), rounds=roundNum)
    