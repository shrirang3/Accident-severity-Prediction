import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from predictions import get_prediction, ordinal_encoding

model = joblib.load("./Notebook/extree_tuned_final.joblib")

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_gender=['Male', 'Female', 'Unknown']
options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

options_realation=['Employee', 'Unknown', 'Owner',  'Other']

options_owner=['Owner', 'Governmental', 'Organization', 'Other']

options_service_year=['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown', 'Below 1yr']

options_education=['Above high school', 'Junior high school', 'Elementary school',
 'High school', 'Unknown', 'Illiterate', 'Writing & reading']

options_defect=['No defect', '7', '5']

options_road_all=['Tangent road with flat terrain', 
 'Tangent road with mild grade and flat terrain', 'Escarpments',
 'Tangent road with rolling terrain', 'Gentle horizontal curve',
 'Tangent road with mountainous terrain and',
 'Steep grade downward with mountainous terrain', 'Sharp reverse curve',
 'Steep grade upward with mountainous terrain']

options_junctions=['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
 'X Shape']

options_surface_type=['Asphalt roads', 'Earth roads' , 'Asphalt roads with some distress',
 'Gravel roads', 'Other']

options_road_surfcondition=['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']

options_light_conditions=['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
 'Darkness - lights unlit']

options_weather_conditions=['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow',
 'Unknown', 'Fog or mist']

options_typeofcollision=['Collision with roadside-parked vehicles',
 'Vehicle with vehicle collision', 'Collision with roadside objects',
 'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles',
 'Collision with pedestrians', 'With Train', 'Unknown']

options_vehicle_movement=['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go',
 'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking',
 'Other', 'Entering a junction']

options_causuality_class=['Driver or rider', 'Pedestrian', 'Passenger']

options_casualty_gender=['Male', 'Female', 'na']

options_casualty_gender=['na', '31-50', '18-30', 'Under 18', 'Over 51', '5']


options_pedestrian_movement=['Not a Pedestrian', "Crossing from driver's nearside",
 'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
 'Unknown or other',
 'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
 'Walking along in carriageway, back to traffic',
 'Walking along in carriageway, facing traffic',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle']

options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']

features = ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")


        age_band_driver=st.selectbox("Select Age Band of Driver: ", options=options_age)
        casualties = st.slider("Number of Casualities: ", 1, 8, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        vehicles_involved = st.slider("Vehicles involved: ", 1, 7, value=0, format="%d")
        light_conditions=st.selectbox("Select Light Conditions: ", options=options_light_conditions)
        hour=st.slider("Select Hour: ", 0, 23, value=0, format="%d")
        accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        type_junctions=st.selectbox("Select Type of Junctions: ", options=options_junctions)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        

        submit = st.form_submit_button("Predict")


    if submit:
        age_band_driver = ordinal_encoding(age_band_driver, options_age)
        day_of_week=ordinal_encoding(day_of_week, options_day)
        light_conditions=ordinal_encoding(light_conditions, options_light_conditions)
        accident_cause=ordinal_encoding(accident_cause, options_cause)
        type_junctions=ordinal_encoding(type_junctions, options_junctions)
        driving_experience=ordinal_encoding(driving_experience, options_driver_exp)



        data = np.array([age_band_driver, casualties, day_of_week, 
                        vehicles_involved, 
                         light_conditions, 
                         hour, accident_cause, type_junctions, driving_experience

       ]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()
