from flask import Flask, request, jsonify, session, url_for, redirect, render_template
import joblib
from form import InputForm

# Load the model
classifier_model = joblib.load('saved_models/KNN.pkl')

def make_prediction(model, sample_json):
    # Extract features from the input JSON
    ph = sample_json['ph']
    Hardness = sample_json['Hardness']
    Solids = sample_json['Solids']
    Chloramines = sample_json['Chloramines']
    Sulfate = sample_json['Sulfate']
    Conductivity = sample_json['Conductivity']
    Organic_carbon = sample_json['Organic_carbon']
    Trihalomethanes = sample_json['Trihalomethanes']
    Turbidity = sample_json['Turbidity']

    # Make a prediction
    prediction = model.predict([[ph, Hardness, Solids, Chloramines, Sulfate,
                                 Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

    if prediction[0] == 0:
        prediction = 'Not Potable'
    else:
        prediction = 'Potable'


    return prediction

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if form.validate_on_submit():
        session['ph'] = form.ph.data
        session['Hardness'] = form.Hardness.data
        session['Solids'] = form.Solids.data
        session['Chloramines'] = form.Chloramines.data
        session['Sulfate'] = form.Sulfate.data
        session['Conductivity'] = form.Conductivity.data
        session['Organic_carbon'] = form.Organic_carbon.data
        session['Trihalomethanes'] = form.Trihalomethanes.data
        session['Turbidity'] = form.Turbidity.data
        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    # Ensure session data is not None
    def get_session_data(key, default=0.0):
        return float(session.get(key, default) or default)

    content = {
        'ph': get_session_data('ph'),
        'Hardness': get_session_data('Hardness'),
        'Solids': get_session_data('Solids'),
        'Chloramines': get_session_data('Chloramines'),
        'Sulfate': get_session_data('Sulfate'),
        'Conductivity': get_session_data('Conductivity'),
        'Organic_carbon': get_session_data('Organic_carbon'),
        'Trihalomethanes': get_session_data('Trihalomethanes'),
        'Turbidity': get_session_data('Turbidity')
    }

    results = make_prediction(classifier_model, content)
    return render_template('prediction.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
