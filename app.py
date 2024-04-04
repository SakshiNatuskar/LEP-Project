from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the trained model
try:
    loaded_model = joblib.load('trained_model.joblib')
except FileNotFoundError:
    print("Model file not found. Make sure 'model.pkl' exists.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    try:

        # Extract form data
        
        Gender = int(request.form['gender'])
        Married = int(request.form['status'])
        Dependents = int(request.form['dependents'])
        Education = int(request.form['education'])
        Self_Employed = int(request.form['selfEmployed'])
        ApplicantIncome = int(request.form['applicantIncome'])
        CoapplicantIncome = int(request.form['coapplicantIncome'])
        LoanAmount = int(request.form['loanAmount'])
        Loan_Amount_Term = int(request.form['loanTerm'])
        Credit_History = int(request.form['creditHistory'])
        Property_Area = int(request.form['propertyArea'])

       
        # Prepare input array for prediction
        x_app = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                           ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                           Credit_History, Property_Area]])

        # Make prediction
        prediction = loaded_model.predict(x_app)

        # Return prediction result
        if prediction == 1:
            message = "Congratulations! Your loan application is approved."
        else:
            message = "We regret to inform you that your loan application is not approved."
        return render_template('result.html', prediction=message)
    except (KeyError, ValueError, FileNotFoundError) as e:
        return render_template('error.html', prediction="Error: Invalid input! Please enter valid values.")
        

@app.route('/self', methods=['GET', 'POST'])
def self_loan_application():
    if request.method == 'GET':
        return render_template('self.html')
    elif request.method == 'POST':
        return prediction()
    


@app.route('/predictsheet', methods=['POST'])
def company_loan_application():
    try:
        # Load the trained model
        loaded_model = joblib.load('trained_model.joblib')

        # Read the CSV file from the form
        file = request.files['csvFile']
        df = pd.read_csv(file)

        # Assuming Gender, Married, Education, Self_Employed, Property_Area are column names in your DataFrame

        # Handle missing values and convert categorical variables
        df['Gender'] = df['Gender'].fillna('').apply(lambda x: 1 if str(x).lower() == 'male' else 0)
        df['Married'] = df['Married'].fillna('').apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        df['Education'] = df['Education'].fillna('').apply(lambda x: 0 if str(x).lower() == 'graduate' else 1)
        df['Self_Employed'] = df['Self_Employed'].fillna('').apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

        # Handle Dependents column
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

        # Handle Property_Area transformation with dictionary mapping
        property_area_mapping = {'rural': 0, 'semiurban': 1, 'urban': 2}
        df['Property_Area'] = df['Property_Area'].fillna('').apply(lambda x: property_area_mapping.get(str(x).lower(), 0))

        # Prepare data for prediction
        X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
        df['Loan_Status_Predicted'] = loaded_model.predict(X)

        # Convert predicted labels to readable format
        df['Loan_Status_Predicted'] = df['Loan_Status_Predicted'].apply(lambda x: 'Approved' if x == 1 else 'Not Approved')

        # Save the processed Excel file
        output_filename = 'processed_loan_data.xlsx'
        df.to_excel(output_filename, index=False)

        # Return the processed file to the user for download
        return send_file(output_filename, as_attachment=True)
    except FileNotFoundError:
        return render_template('error.html', message='Model file not found.')


@app.route('/static/display_code', methods=['GET'])
def display_code():
    # Read the Python files containing the Flask application and HTML templates
    with open('app.py', 'r') as app_file:
        app_code = app_file.read()

    with open('static/company.html', 'r') as company_template_file:
        company_template_code = company_template_file.read()

    with open('static/display_code.html', 'r') as display_template_file:
        display_template_code = display_template_file.read()

    # Pass the code to the template for rendering
    return render_template('static/display_code.html', app_code=app_code, company_template_code=company_template_code, display_template_code=display_template_code)

app.run(debug=True)
