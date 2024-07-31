from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField


class InputForm(FlaskForm):
    """
    Form to take input from user
    
    """

    ph = StringField('pH')
    Hardness = StringField('Hardness')
    Solids = StringField('Solids')
    Chloramines = StringField('Chloramines')
    Sulfate = StringField('Sulfate')
    Conductivity = StringField('Conductivity')
    Organic_carbon = StringField('Organic Carbon')
    Trihalomethanes = StringField('Trihalomethanes')
    Turbidity = StringField('Turbidity')
    submit = SubmitField('Submit')