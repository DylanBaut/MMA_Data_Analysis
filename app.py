from flask import Flask, render_template, request, flash, redirect, url_for, session
from torch import load, no_grad, tensor, float32
from torch.nn import Sequential, Linear, ReLU
app = Flask(__name__)
app.secret_key="jaxb"
formData ={}

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