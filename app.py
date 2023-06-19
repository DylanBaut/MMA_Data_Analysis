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
    for rowtuple in scores_df.itertuples():
        row= list(rowtuple)[1:]
        temp = row[0] + " vs. " + row[1]
        flash(temp, "fight")
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
    for rowtuple in scores_df.itertuples():
        row= list(rowtuple)[1:]
        temp = row[0] + " vs. " + row[1]
        flash(temp, "fight")
    row = scores_df.iloc[int(formData['fightselect'])]
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
    for tier in range(len(results)):
        result =results[tier]
        indexMin = result.index(min(result))
        minVal = result[indexMin]
        for idx in range(len(result)):
            result[idx]= result[idx]-minVal
        sumVal = sum(result)
        for idx in range(len(result)):
            result[idx]= round(((result[idx]/sumVal)*100),2)
            flash(result[idx], tier)
    print(results)
    if request.method =='POST':
        formData['fightselect']=request.form['fight-list']
        print(formData['fightselect'])
        return redirect(url_for('searchOutput'))
    else:
        return render_template("search2.html",len =len(scores_df), rounds=roundNum)