import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Define the prediction function
def predict_diabetes(inputs):
    # Convert inputs to numpy array and reshape for prediction
    inputs_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(inputs_array)
    prediction_prob = model.predict_proba(inputs_array)
    return prediction, prediction_prob

# Define the function for healthcare recommendations
def provide_healthcare_recommendations(prediction, user_inputs):
    recommendations = []
    
    if prediction[0] == 1:
        recommendations.append("Since the model predicts that you have diabetes, it is important to follow a diabetes management plan. Here are some recommendations:")
    else:
        recommendations.append("Since the model predicts that you do not have diabetes, here are some general health recommendations to help prevent diabetes:")
    
    # General health recommendations
    if user_inputs[1] > 140:  # High glucose
        recommendations.append("- Monitor your blood sugar levels regularly.")
        recommendations.append("- Reduce your intake of sugary foods and drinks.")
        recommendations.append("- Consult with a healthcare provider for personalized advice.")
    
    if user_inputs[2] > 80:  # High blood pressure
        recommendations.append("- Maintain a healthy blood pressure through regular exercise and a balanced diet.")
        recommendations.append("- Reduce salt intake and avoid processed foods.")
    
    if user_inputs[5] > 25:  # High BMI
        recommendations.append("- Aim for a healthy weight through diet and exercise.")
        recommendations.append("- Incorporate more fruits, vegetables, and whole grains into your diet.")
    
    if user_inputs[7] > 45:  # Age over 45
        recommendations.append("- Schedule regular check-ups with your healthcare provider.")
        recommendations.append("- Stay physically active to maintain overall health.")
    
    # General diabetes management recommendations
    recommendations.append("- Exercise regularly, aiming for at least 30 minutes of moderate activity most days.")
    recommendations.append("- Follow a balanced diet rich in fiber, and low in refined sugars and saturated fats.")
    recommendations.append("- Stay hydrated and avoid sugary beverages.")
    recommendations.append("- Monitor your blood sugar levels as recommended by your healthcare provider.")
    recommendations.append("- Take any prescribed medications as directed.")

    return recommendations

# Streamlit UI
st.title("DIABETES PREDICTION MODEL BY SIDDHARTH KUMKALE")
st.markdown("""
<style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for input fields
st.sidebar.header("Enter Patient Data")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=110)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=846, value=79)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0, format="%.1f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42, value=0.5, format="%.2f")
age = st.sidebar.number_input("Age", min_value=21, max_value=81, value=33)

# Collect the inputs into an array
user_inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# Predict diabetes
if st.sidebar.button("Predict"):
    prediction, prediction_prob = predict_diabetes(user_inputs)
    
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.write("The model predicts that the patient **has diabetes**.")
    else:
        st.write("The model predicts that the patient **does not have diabetes**.")
    
    st.write(f"Probability of having diabetes: {prediction_prob[0][1]:.2f}")
    st.write(f"Probability of not having diabetes: {prediction_prob[0][0]:.2f}")
    
    # Visualize the prediction probabilities using Plotly
    fig = go.Figure(go.Bar(
        x=[prediction_prob[0][0], prediction_prob[0][1]],
        y=['No Diabetes', 'Diabetes'],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability",
        yaxis_title="Condition",
        xaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig)
    
    # Provide healthcare recommendations
    st.subheader("Healthcare Recommendations")
    recommendations = provide_healthcare_recommendations(prediction, user_inputs)
    for recommendation in recommendations:
        st.write(recommendation)

# To run the app, use the command: streamlit run app.py
