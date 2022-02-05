import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
loadedModel = pickle.load(open('KNN Model.pkl', 'rb'))

#WWW.google.co.in/prediction

#Routes
@app.route('/')
def home():
    return render_template('cancer.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    Mean_radius = request.form['Mean_radius']
    Mean_Texture = request.form['Mean_texture']
    Mean_perimeter = request.form['Mean_perimeter']
    Mean_area = request.form['Mean_area']
    Mean_smoothness =request.form['Mean_smoothness']

    prediction = loadedModel.predict([[Mean_radius, Mean_Texture,  Mean_perimeter, Mean_area, Mean_smoothness]])[0]

    if prediction == 0:
        prediction = "Benign"
    else:
        prediction = "Malignant"

    return render_template('cancer.html', output=prediction)

#Main function
if __name__ == '__main__':
     app.run(debug=True)   