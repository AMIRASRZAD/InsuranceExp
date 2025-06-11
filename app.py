from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2 import pool
import uuid
import json
import random
import time
import logging
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import skewnorm

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', format='%(asctime)s %(levelname)s: %(message)s')

# Neon connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    host=os.environ.get('DB_HOST', 'ep-odd-boat-a5tpi1i2-pooler.us-east-2.aws.neon.tech'),
    port=os.environ.get('DB_PORT', 5432),
    database=os.environ.get('DB_NAME', 'neondb'),
    user=os.environ.get('DB_USER', 'neondb_owner'),
    password=os.environ.get('DB_PASSWORD', 'REDACTED'),
    sslmode='require'
)
'''
# Load CSV (assumed uploaded, using provided schema)
# Placeholder: Replace with actual CSV path after upload
df = pd.DataFrame({
    'age': [31, 60, 31, 18, 33],
    'sex': ['female', 'male', 'male', 'female', 'female'],
    'bmi': [38.095, 32.8, 29.81, 38.665, 35.53],
    'children': [1, 0, 0, 2, 0],
    'smoker': ['yes', 'yes', 'yes', 'no', 'yes'],
    'region': ['northeast', 'southwest', 'southeast', 'northeast', 'northwest'],
    'true_charges': [58571, 52591, 19350, 3393, 55135],
    'predicted_charges': [42710, 40164, 27534, 8583, 38827],
    'prediction_error': [15862, 12427, 8183, 5190, 16309],
    'total_uncertainty_std': [3830, 5892, 6530, 10384, 4454],
    'epistemic_uncertainty_std': [1211, 1065, 996, 1294, 798],
    'aleatoric_uncertainty_std': [3634, 5795, 6453, 10303, 4382],
    'uncertainty_level': [1, 1, 1, 1, 2],
    'condition': ['epistemic', 'epistemic', 'epistemic', 'epistemic', 'epistemic']
})
'''
CSV_URL = "https://drive.google.com/uc?id=1z_FUKc-_5n3Z5gqTbWjgqI4HETHzaO4zP"
df = pd.read_csv(CSV_URL)

for col in ['age', 'children']:
    df[col] = df[col].astype(int)
for col in ['bmi', 'true_charges', 'predicted_charges', 'prediction_error', 'total_uncertainty_std', 'epistemic_uncertainty_std', 'aleatoric_uncertainty_std']:
    df[col] = df[col].astype(float)
df['sex_enc'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_enc'] = df['smoker'].map({'yes': 1, 'no': 0})
region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
df['region_enc'] = df['region'].map(region_map)

# Define averages
AVERAGES = {
    'age': 39,
    'bmi': 30,
    'children': 1.1,
    'charges': 13270
}

# Participant counts for conditions (1, 2, 3)
PARTICIPANT_COUNTS = {1: 0, 2: 0, 3: 0}
MAX_PER_CONDITION = 80  # For 240 users
PARTICIPANT_ORDER = []

def epistemic_charts(uncertainty_level, task_region):
    regions = ['northeast', 'northwest', 'southeast', 'southwest']
    task_region_idx = regions.index(task_region)
    volumes = [0] * 4  # 0=low, 1=moderate, 2=high
    if uncertainty_level == 1:
        volumes = [1, 1, 0, 0]
        if task_region_idx in [2, 3]:
            volumes[task_region_idx], volumes[2 if task_region_idx == 3 else 3] = 0, 0
        else:
            volumes[task_region_idx], volumes[0 if task_region_idx == 1 else 1] = 0, 0
    elif uncertainty_level == 2:
        volumes = [1, 1, 0, 0]
        volumes[task_region_idx] = 1
        other_moderate = (task_region_idx + 1) % 4
        volumes[other_moderate] = 1
    elif uncertainty_level == 3:
        volumes = [1, 1, 2, 2]
        volumes[task_region_idx] = 1
        other_moderate = (task_region_idx + 1) % 4
        volumes[other_moderate] = 1
    elif uncertainty_level == 4:
        volumes = [2, 2, 2, 1]
        volumes[task_region_idx] = 2
        other_moderate = (task_region_idx + 1) % 4
        volumes[other_moderate] = 1

    datasets = [{
        'label': 'Data Volume',
        'data': volumes,
        'backgroundColor': ['#FF6B6B' if v == 0 else '#FFD93D' if v == 1 else '#4CAF50' for v in volumes],
        'borderColor': ['#D32F2F' if v == 0 else '#FBC02D' if v == 1 else '#388E3C' for v in volumes],
        'borderWidth': 1
    }]

    return {
        'type': 'bar',
        'data': {
            'labels': regions,
            'datasets': datasets
        },
        'options': {
            'scales': {
                'y': {'beginAtZero': True, 'max': 2.5, 'ticks': {'callback': 'function(value) { return ["Low", "Moderate", "High"][value]; }'}},
                'x': {'title': {'display': True, 'text': 'Region'}}
            },
            'plugins': {
                'legend': {'display': False},
                'title': {'display': True, 'text': 'AI Training Data Volume by Region'}
            }
        }
    }

def aleatoric_charts(level, predicted_charge, error):
    skewness = random.uniform(0.5, 1.5) if level in [1, 2] else random.uniform(0, 0.5)
    scale = 10000 if level == 1 else 7500 if level == 2 else 5000 if level == 3 else 2500
    error_range = abs(error) * 1.5
    x = np.linspace(predicted_charge - error_range, predicted_charge + error_range * 1.5, 100)
    y = skewnorm.pdf(x, skewness, loc=predicted_charge, scale=scale)
    y = y / y.max() * (5 if level == 1 else 6 if level == 2 else 7 if level == 3 else 8)

    plt.figure(figsize=(6, 3))
    plt.plot(x, y, color='#4CAF50', label='Predictive Distribution')
    plt.fill_between(x, y, alpha=0.3, color='#4CAF50')
    plt.axvline(predicted_charge, color='#D32F2F', linestyle='--', label='Predicted Charge')
    plt.xlabel('Insurance Charge (USD)')
    plt.ylabel('Density')
    plt.title(f'Predictive Distribution (Uncertainty Level {level})')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def sample_rows():
    sampled_rows = df.sample(20, random_state=np.random.randint(1000)).to_dict('records')
    return sampled_rows

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    available_conditions = [c for c, count in PARTICIPANT_COUNTS.items() if count < MAX_PER_CONDITION]
    if not available_conditions:
        return "Experiment is full!", 403

    # Consecutive assignment: cycle through 1, 2, 3
    condition = (len(PARTICIPANT_ORDER) % 3) + 1
    if condition not in available_conditions:
        return "Condition full, try again later.", 403

    participant_name = request.form.get('participant_name', '').strip() or None
    participant_id = str(uuid.uuid4())

    PARTICIPANT_COUNTS[condition] += 1
    PARTICIPANT_ORDER.append(condition)
    logging.info(f"Assigned participant {participant_id} to condition {condition}")

    session['condition'] = condition
    session['participant_id'] = participant_id
    session['participant_name'] = participant_name
    session['practice_index'] = 0
    session['tasks'] = sample_rows()
    session['task_index'] = 0
    session['responses'] = []

    return redirect(url_for('practice'))

@app.route('/practice', methods=['GET', 'POST'])
def practice():
    practice_index = session.get('practice_index', 0)
    practice_data = df.head(5).to_dict('records')  # First 5 rows for practice
    if practice_index >= len(practice_data):
        return redirect(url_for('task'))

    practice_row = practice_data[practice_index]
    customer_number = practice_index + 1

    if request.method == 'POST':
        initial_guess = request.form.get('initial_guess')
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 100000:
                raise ValueError
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid charge guess. Please select a value between 1 and 100,000 USD.", 400

        session['practice_index'] += 1
        return render_template('practice_result.html',
                             customer_number=customer_number,
                             customer_info=practice_row,
                             initial_guess=initial_guess,
                             true_charge=practice_row['true_charges'])

    customer_info = {
        'age': practice_row['age'],
        'sex': practice_row['sex'],
        'bmi': practice_row['bmi'],
        'children': practice_row['children'],
        'smoker': practice_row['smoker'],
        'region': practice_row['region']
    }
    return render_template('practice.html',
                         customer_info=customer_info,
                         customer_number=customer_number,
                         averages=AVERAGES)

@app.route('/task', methods=['GET', 'POST'])
def task():
    task_index = session.get('task_index', 0)
    if task_index >= len(session.get('tasks', [])):
        conn = None
        try:
            conn = db_pool.getconn()
            with conn.cursor() as cur:
                for response in session.get('responses', []):
                    task_data = next(t for t in session['tasks'] if t['ID'] == response['ID'])
                    cur.execute(
                        "INSERT INTO responses (participant_id, task_number, condition, initial_guess, final_guess, predicted_charge, uncertainty_level, "
                        "true_charge, age, sex, bmi, children, smoker, region, prediction_error, total_uncertainty_std, epistemic_uncertainty_std, aleatoric_uncertainty_std) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (session['participant_id'], response['Task_Number'], response['Condition'], response['Initial_Guess'],
                         response['Final_Guess'], response['Predicted_Charge'], response['Uncertainty_Level'],
                         task_data['true_charges'], task_data['age'], task_data['sex'], task_data['bmi'],
                         task_data['children'], task_data['smoker'], task_data['region'], task_data['prediction_error'],
                         task_data['total_uncertainty_std'], task_data['epistemic_uncertainty_std'], task_data['aleatoric_uncertainty_std'])
                    )
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Database error: {str(e)}")
            return "Error saving data", 500
        finally:
            if conn:
                db_pool.putconn(conn)
        return render_template('end.html')

    task_data = session['tasks'][task_index]
    condition = session['condition']
    customer_number = task_index + 1

    if request.method == 'POST':
        initial_guess = request.form.get('initial_guess')
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 100000:
                raise ValueError
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid charge guess. Please select a value between 1 and 100,000 USD.", 400
        session['current_initial_guess'] = initial_guess
        return redirect(url_for('stage2'))

    customer_info = {
        'age': task_data['age'],
        'sex': task_data['sex'],
        'bmi': task_data['bmi'],
        'children': task_data['children'],
        'smoker': task_data['smoker'],
        'region': task_data['region']
    }
    return render_template('assessment.html',
                         customer_info=customer_info,
                         averages=AVERAGES,
                         customer_number=customer_number)

@app.route('/stage2', methods=['GET', 'POST'])
def stage2():
    task_index = session.get('task_index', 0)
    task_data = session['tasks'][task_index]
    condition = session['condition']
    initial_guess = session.get('current_initial_guess')
    customer_number = task_index + 1

    if request.method == 'POST':
        if condition == 1 or 'show_ai_info' in session:
            final_guess = request.form.get('final_guess')
            try:
                final_guess = float(final_guess)
                if final_guess < 1 or final_guess > 100000:
                    raise ValueError
            except (ValueError, TypeError):
                logging.error(f"Invalid final_guess: {final_guess}")
                return "Invalid charge guess. Please select a value between 1 and 100,000 USD.", 400
            session['current_final_guess'] = final_guess
            return redirect(url_for('stage3'))
        else:
            session['show_ai_info'] = True

    predicted_charge = task_data['predicted_charges']
    info_data = None
    if condition == 2:
        info_data = json.dumps(epistemic_charts(task_data['uncertainty_level'], task_data['region']))
    elif condition == 3:
        info_data = aleatoric_charts(task_data['uncertainty_level'], predicted_charge, task_data['prediction_error'])

    return render_template(f'review_condition{condition}.html',
                         predicted_charge=predicted_charge,
                         initial_guess=initial_guess,
                         show_ai_info=session.get('show_ai_info', False),
                         info_data=info_data,
                         customer_number=customer_number)

@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    task_index = session.get('task_index', 0)
    task_data = session['tasks'][task_index]
    initial_guess = session.get('current_initial_guess')
    final_guess = session.get('current_final_guess')
    customer_number = task_index + 1

    if request.method == 'POST':
        session['responses'].append({
            'ID': task_data['ID'],
            'Task_Number': task_index + 1,
            'Condition': session['condition'],
            'Initial_Guess': initial_guess,
            'Final_Guess': final_guess,
            'Predicted_Charge': task_data['predicted_charges'],
            'Uncertainty_Level': task_data['uncertainty_level'],
            'True_Charge': task_data['true_charges']
        })
        session['task_index'] += 1
        session.pop('show_ai_info', None)
        return redirect(url_for('task'))

    return render_template('results.html',
                         customer_number=customer_number,
                         initial_guess=initial_guess,
                         final_guess=final_guess,
                         ai_prediction=task_data['predicted_charges'])

@app.route('/test-db')
def test_db():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT NOW();")
            result = cur.fetchone()
            logging.debug(f"Test DB successful: {result[0]}")
            return f"Database time: {result[0]}"
    except Exception as e:
        logging.error(f"Test DB failed: {str(e)}")
        return f"Connection failed: {e}"
    finally:
        if conn:
            db_pool.putconn(conn)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)