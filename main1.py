from flask import Flask , render_template , request
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt,mpld3
import requests
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
@app.route('/sys.html')
def sys():
    return render_template('sym.html')
@app.route('/table')
def Table():
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
    state_data['Confirmed'] = state_data['Confirmed'].map(int)
    state_data['Recovered'] = state_data['Recovered'].map(int)
    state_data['Deceased'] = state_data['Deceased'].map(int)
    table = PrettyTable()
    table.field_names = (new_cols)
    for i in stats:
        table.add_row(i)
        table.add_row(["","Total", 
               sum(state_data['Confirmed']), 
               sum(state_data['Recovered']),
               sum(state_data['Deceased'])])
    sns.set_style("ticks")
    plt.figure(figsize = (15,10))
    plt.barh(state_data["States/UT"],    state_data["Confirmed"].map(int),align = 'center', color = 'lightblue', edgecolor = 'blue')
    plt.xlabel('No. of Confirmed cases', fontsize = 18)
    plt.ylabel('States/UT', fontsize = 18)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title('Total Confirmed Cases Statewise', fontsize = 18 )
    for index, value in enumerate(state_data["Confirmed"]):
        plt.text(value, index, str(value), fontsize = 12)
    plt.savefig('/static/images/new_plot.png')
    return render_template('sym.html',pf='/static/images/new_plot.png')
if __name__ == "__main__":
    app.run(debug=True)