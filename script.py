#importing libraries
import os
import numpy as np
import flask
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
        return flask.render_template('index.html')

def ValuePredictor(val):
        values_to_predict = np.array(val).reshape(1, 151)
        model = joblib.load("trained_suicide_rate_model.pkl")
        result = model.predict(values_to_predict)
        return result[0]

@app.route('/result',methods = ['POST'])
def result():
        if request.method == 'POST':
                df = pd.DataFrame(columns=['population', 'HDI for year', 'gdp_for_year', 'gdp_per_capita', 'age_15-24 years', 'age_25-34 years', 'age_35-54 years', 'age_5-14 years', 'age_55-74 years', 'age_75+ years', 'sex_female', 'sex_male', 'generation_Boomers', 'generation_G.I. Generation', 'generation_Generation X', 'generation_Generation Z', 'generation_Millenials', 'generation_Silent',  'country_Albania', 'country_Antigua and Barbuda', 'country_Argentina', 'country_Armenia', 'country_Aruba', 'country_Australia', 'country_Austria', 'country_Azerbaijan', 'country_Bahamas', 'country_Bahrain', 'country_Barbados', 'country_Belarus', 'country_Belgium', 'country_Belize', 'country_Bosnia and Herzegovina', 'country_Brazil', 'country_Bulgaria', 'country_Cabo Verde', 'country_Canada', 'country_Chile', 'country_Colombia', 'country_Costa Rica', 'country_Croatia', 'country_Cuba', 'country_Cyprus', 'country_Czech Republic', 'country_Denmark', 'country_Dominica', 'country_Ecuador', 'country_El Salvador', 'country_Estonia', 'country_Fiji', 'country_Finland', 'country_France', 'country_Georgia', 'country_Germany', 'country_Greece', 'country_Grenada', 'country_Guatemala', 'country_Guyana', 'country_Hungary', 'country_Iceland', 'country_Ireland', 'country_Israel', 'country_Italy', 'country_Jamaica', 'country_Japan', 'country_Kazakhstan', 'country_Kiribati', 'country_Kuwait', 'country_Kyrgyzstan', 'country_Latvia', 'country_Lithuania', 'country_Luxembourg',
                                           'country_Macau', 'country_Maldives', 'country_Malta', 'country_Mauritius', 'country_Mexico', 'country_Mongolia', 'country_Montenegro', 'country_Netherlands', 'country_New Zealand', 'country_Nicaragua', 'country_Norway', 'country_Oman', 'country_Panama', 'country_Paraguay', 'country_Philippines', 'country_Poland', 'country_Portugal', 'country_Puerto Rico', 'country_Qatar', 'country_Republic of Korea', 'country_Romania', 'country_Russian Federation', 'country_Saint Kitts and Nevis', 'country_Saint Lucia', 'country_Saint Vincent and Grenadines', 'country_San Marino', 'country_Serbia', 'country_Seychelles', 'country_Singapore', 'country_Slovakia', 'country_Slovenia', 'country_South Africa', 'country_Spain', 'country_Sri Lanka', 'country_Suriname', 'country_Sweden', 'country_Switzerland', 'country_Thailand', 'country_Trinidad and Tobago', 'country_Turkey', 'country_Turkmenistan', 'country_Ukraine', 'country_United Arab Emirates', 'country_United Kingdom', 'country_United States', 'country_Uruguay', 'country_Uzbekistan', 'year_1985', 'year_1986', 'year_1987', 'year_1988', 'year_1989', 'year_1990', 'year_1991', 'year_1992', 'year_1993', 'year_1994', 'year_1995', 'year_1996', 'year_1997', 'year_1998', 'year_1999', 'year_2000', 'year_2001', 'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016'])

                changeValues = [request.form['population'], int(request.form["hdi"]), request.form["gdp-year"],
                request.form["gdp-capita"], 1, 1, 1, 1, 1]
                changeColumns = ['population', 'HDI for year', 'gdp_for_year', 'gdp_per_capita', request.form['age'], request.form['sex'],
                request.form['generation'], request.form['country'], request.form['year']]
                
                df_new = pd.DataFrame([changeValues], columns = changeColumns)

                final = df_new.append(df)[df.columns.tolist()]
                final = final.fillna(0)
                
                final.reindex(df, axis=1)
                results = ValuePredictor(final)
                #This has a accuracy of 97% so i'd consider it a success
                prediction = 'The predicted suicide rate is {}'.format(results)

                #General classification of suicide rates (based off WHO's data)
                if (results < 5.0):
                        suicide_rate_classification = 'Low'
                        icon = 'fa-ambulance'
                        number = '1'
                elif ((results >= 5.0) and (results < 10.0)):
                        suicide_rate_classification = 'Medium'
                        icon = 'fa-ambulance'
                        number = '2'
                elif ((results >= 10.0) and (results < 15.0)):
                        suicide_rate_classification = 'High'
                        icon = 'fa-ambulance'
                        number = '3'
                elif (results >= 15.0):
                        suicide_rate_classification = 'Extremely-High'
                        icon = 'fa-ambulance'
                        number = '4'
                else:
                        suicide_rate_classification = 'Not-Applicable'
                        icon = 'fa-question'
                        number = '1'
                
                return render_template("result.html", prediction=prediction, rate=suicide_rate_classification, icon=icon, number=number)
