from flask import Flask , render_template , request
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt,mpld3
import requests
from tabulate import tabulate
from bs4 import BeautifulSoup
from prettytable import PrettyTable
PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

#clf load krenge
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method =="POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        #ye ouput ka mamla h
        inputfeatures=[fever,pain,age,runnyNose,diffBreath]
        infprob=clf.predict_proba([inputfeatures])[0][1]
        print(infprob)
        return render_template('show.html', inf=round(infprob*100))
    return render_template('index.html')
@app.route('/symptoms')
def symptoms():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'symptoms.png')
    return render_template('sympotms.html', user_image = full_filename)
@app.route('/covid')
def covid():
    return render_template('covid.html')
@app.route('/table')
def table():
    url = 'https://www.mohfw.gov.in/'
# make a GET request to fetch the raw HTML content
    web_content = requests.get(url).content
# parse the html content
    soup = BeautifulSoup(web_content, "html.parser")
# remove any newlines and extra spaces from left and right
    extract_contents = lambda row: [x.text.replace('\n', '') for x in row]
# find all table rows and data cells within
    stats = [] 
    all_rows = soup.find_all('tr')
    for row in all_rows:
        stat = extract_contents(row.find_all('td')) 
# notice that the data that we require is now a list of length 5
        if len(stat) == 5:
            stats.append(stat)
#now convert the data into a pandas dataframe for further processing
    new_cols = ["Sr.No", "States/UT","Confirmed","Recovered","Deceased"]
    state_data = pd.DataFrame(data = stats, columns = new_cols)
    a=tabulate(state_data,new_cols,tablefmt="github",numalign="right")
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'india.png')
    return render_template('table.html', user_image = full_filename)
@app.route('/pre')
def pre():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'corona.png')
    full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'corona2.jpeg')
    return render_template('precaution.html', user_image = full_filename , ff=full_filename1)
if __name__ == "__main__":
    app.run(debug=True)