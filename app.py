import streamlit as st
import pickle
import numpy as np

# Load models
logistic = pickle.load(open("logistic.pkl", "rb"))
knn = pickle.load(open("knn.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))

# Load accuracy
accuracy_dict = pickle.load(open("accuracy.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🫁 Lung Cancer Prediction System")
st.write("Select algorithm and enter patient details")

# Dropdown
model_option = st.selectbox(
    "Choose Algorithm",
    ["Logistic Regression", "SVM", "KNN"]
)

st.info(f"Model Accuracy: {round(accuracy_dict[model_option]*100,2)}%")

# Inputs
# age = st.slider("Select your Age", 1, 100, 30)
# gender = st.selectbox("Gender", ["Male", "Female"])
# smoking = st.selectbox("Smoking", ["No", "Yes"])
# fing_discolor = st.selectbox("Finger Discoloration", ["No", "Yes"])
# mentalstress = st.selectbox("Mental Stress", ["No", "Yes"])
# exposure_to_pollution = st.selectbox("Exposure to Pollution", ["No" , "Yes"])
# long_term_illness = st.selectbox("Long Term Illness", ["No" , "Yes"])
# energy_level = st.number_input("Energy Level:", value=0.0)
# immune_weakness = st.selectbox("Immune Weakness",["No","Yes"])
# breathing_issue = st.selectbox("Breathing Issue",["No","Yes"])
# alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
# throat_discomfort = st.selectbox("Throat Discomfort", ["No", "Yes"])
# oxygen_saturation = st.number_input("Oxygen Saturation:", value=0.0)
# chest_tightness = st.selectbox("Chest Tightness",["No","Yes"])
# family_history = st.selectbox("Family History",["No","Yes"])
# smoking_family_history = st.selectbox("Smoking Family History",["No","Yes"])
# stress_immune = st.selectbox("Stress Immune",["No","Yes"])



# Inputs
age = st.slider("Select Your Age", 1, 100, 30)

gender = st.selectbox("Select Your Gender", ["Male", "Female"])

smoking = st.selectbox("Do You Smoke?", ["No", "Yes"])

fing_discolor = st.selectbox("Do You Have Finger Discoloration (Yellowing of Fingers)?", ["No", "Yes"])

mentalstress = st.selectbox("Do You Frequently Experience Mental Stress?", ["No", "Yes"])

exposure_to_pollution = st.selectbox("Are You Regularly Exposed to Air Pollution?", ["No", "Yes"])

long_term_illness = st.selectbox("Do You Have Any Long-Term Illness?", ["No", "Yes"])

energy_level = st.number_input("Enter Your Energy Level (40 = Very Low, 70 = Very High)", min_value=40.0, max_value=70.0, value=50.0)

immune_weakness = st.selectbox("Do You Have a Weak Immune System?", ["No", "Yes"])

breathing_issue = st.selectbox("Do You Experience Breathing Problems?", ["No", "Yes"])

alcohol = st.selectbox("Do You Consume Alcohol?", ["No", "Yes"])

throat_discomfort = st.selectbox("Do You Have Persistent Throat Discomfort?", ["No", "Yes"])

oxygen_saturation = st.number_input("Enter Your Oxygen Saturation Level (SpO2 %)", min_value=0.0, max_value=100.0, value=98.0)

chest_tightness = st.selectbox("Do You Feel Chest Tightness?", ["No", "Yes"])

family_history = st.selectbox("Is There a History of Lung Cancer in Your Family?", ["No", "Yes"])

smoking_family_history = st.selectbox("Do Family Members Smoke Regularly?", ["No", "Yes"])

stress_immune = st.selectbox("Has Stress Affected Your Immunity?", ["No", "Yes"])


# Encoding function
def encode(value):
    return 1 if value == "Yes" else 0

# Encode gender separately
gender_encoded = 1 if gender == "Male" else 0

# Create input array (MAKE SURE ORDER MATCHES TRAINING DATASET)
input_data = np.array([[ 
    age,
    gender_encoded,
    encode(smoking),
    encode(fing_discolor),
    encode(mentalstress),
    encode(exposure_to_pollution),
    encode(long_term_illness),
    energy_level,
    encode(immune_weakness),
    encode(breathing_issue),
    encode(alcohol),
    encode(throat_discomfort),
    oxygen_saturation,
    encode(chest_tightness),
    encode(family_history),
    encode(smoking_family_history),
    encode(stress_immune)
]])

# Scale
input_data = scaler.transform(input_data)

# Select model
if model_option == "Logistic Regression":
    model = logistic
elif model_option == "SVM":
    model = svm
else:
    model = knn

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:

        st.success("✅ Low Risk of Lung Cancer")
