from flask import Flask, request, jsonify , session ,  url_for , redirect , render_template
import joblib

from form import InputForm


classifier_model = joblib.load('saved_models/CatBoostModel.pkl')


def prediction(model, sample_json):

    # For features
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

    return prediction


app = Flask(__name__)
app.config["SECRET_KEY"]="mysecretkey"


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
    content = {}
    content['ph'] = float(session['ph'])
    content['Hardness'] = float(session['Hardness'])
    content['Solids'] = float(session['Solids'])
    content['Chloramines'] = float(session['Chloramines'])
    content['Sulfate'] = float(session['Sulfate'])
    content['Conductivity'] = float(session['Conductivity'])
    content['Organic_carbon'] = float(session['Organic_carbon'])
    content['Trihalomethanes'] = float(session['Trihalomethanes'])
    content['Turbidity'] = float(session['Turbidity'])

    results = prediction(classifier_model, content)

    return render_template('prediction.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)