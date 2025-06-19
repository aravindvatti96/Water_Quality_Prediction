🌊 Water Quality Prediction using Machine Learning
This project predicts multiple water quality parameters using machine learning, specifically a MultiOutputRegressor with RandomForestRegressor. It was developed as part of a one-month AICTE Virtual Internship sponsored by Shell in June 2025.

📘 Overview
Access to clean water is a global necessity. This project aims to assist environmental monitoring by predicting critical water quality indicators from historical data using supervised machine learning.

🔍 Project Highlights
✅ Preprocessed real-world water quality datasets (2000–2021)

✅ Built a multi-output regression model

✅ Visualized feature importance and evaluation metrics

✅ Included custom input for real-time prediction

📊 Predicted Water Quality Parameters
The model predicts the following:

NH4 – Ammonium

BSK5 (BOD5) – Biological Oxygen Demand

Suspended Solids

O2 – Dissolved Oxygen

NO3 – Nitrates

NO2 – Nitrites

SO4 – Sulfates

PO4 – Phosphates

CL – Chlorides

🧠 Model and Evaluation
Model: MultiOutputRegressor (Random Forest)

Evaluation:

R² Score

Mean Squared Error (MSE)

Performance showed strong predictive accuracy, especially for parameters like SO4 and CL.

🛠 Technologies Used
Tool	Purpose
Python 3.12	Core programming
Pandas, NumPy	Data manipulation
Scikit-learn	Machine learning
Matplotlib, Seaborn	Visualization
Streamlit / Jupyter	UI & interactive notebooks

📅 Internship Details
Internship: AICTE Virtual Internship

Organization: Edunet Foundation

Sponsor: Shell

Duration: June 2025 (1 Month)

Focus: Environmental Monitoring using ML

🚀 Run Locally
Requirements:
bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
To run the app:
bash
Copy
Edit
streamlit run app.py
📂 Folder Structure
bash
Copy
Edit
├── app.py                  # Streamlit app
├── PB_All_2000_2021.csv    # Dataset
├── Water_Quality_Pred.ipynb # Notebook version
├── README.md
🤝 Contributions
Feel free to fork this repo, raise issues, or submit pull requests to improve the dataset or modeling techniques!

