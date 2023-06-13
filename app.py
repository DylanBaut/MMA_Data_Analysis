from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)

@app.route("/", methods =['POST', 'GET'])
def index():
    if request.method =='POST':
        sigAtt1=request.form['sigAtt1']
        sigLand1=request.form['sigLand1']
        KD1=request.form['KD1']
        TD1=request.form['TD1']
        sub1=request.form['sub1']
        ctrl1=request.form['ctrl1']
        head1=request.form['head1']
        sigAtt2=request.form['sigAtt2']
        sigLand2=request.form['sigLand2']
        KD2=request.form['KD2']
        TD2=request.form['TD2']
        sub2=request.form['sub2']
        ctrl2=request.form['ctrl2']
        head2=request.form['head2']
        return redirect(url_for('output'))
    else:
        return render_template("index.html")
'''
def main():
    print('x')
if __name__ == "__main__":
    main()
    '''