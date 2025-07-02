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
import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

logging.basicConfig(level=logging.DEBUG, filename='app.log', format='%(asctime)s %(levelname)s: %(message)s')

db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    host=os.environ.get('DB_HOST'),
    port=os.environ.get('DB_PORT', '5432'),
    database=os.environ.get('DB_NAME'),
    user=os.environ.get('DB_USER'),
    password=os.environ.get('DB_PASSWORD'),
    sslmode='require',
    channel_binding='require'
)

CSV_URL = "https://drive.google.com/uc?id=1z_FUKc-_5n3Z5gqTbWjgI4HETHzaO4zP"
df = pd.read_csv(CSV_URL)

for col in ['age', 'children']:
    df[col] = df[col].astype(int)
for col in ['bmi', 'true_charges', 'predicted_charges', 'prediction_error', 'total_uncertainty_std', 'epistemic_uncertainty_std', 'aleatoric_uncertainty_std']:
    df[col] = df[col].astype(float)
df['sex_enc'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_enc'] = df['smoker'].map({'yes': 1, 'no': 0})
region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
df['region_enc'] = df['region'].map(region_map)
df['uncertainty_level'] = df.apply(
    lambda x: 1 if pd.isna(x['uncertainty_level']) or x['uncertainty_level'] == 'NA' or x['condition'] in ['AI_only', 'practice']
    else x['uncertainty_level'], axis=1
)
df['uncertainty_level'] = df['uncertainty_level'].astype(int)

AVERAGES = {
    'age': 39,
    'bmi': 30,
    'charges': 13270
}

PARTICIPANT_COUNTS = {1: 0, 2: 0, 3: 0}
MAX_PER_CONDITION = 80
PARTICIPANT_ORDER = []

def epistemic_charts(uncertainty_level, task_id):
    random.seed(task_id)
    if uncertainty_level == 1:
        percentage = random.uniform(20, 30)
        color = '#FF6B6B'
    elif uncertainty_level == 2:
        percentage = random.uniform(45, 55)
        color = '#FFD93D'
    elif uncertainty_level == 3:
        percentage = random.uniform(65, 75)
        color = '#FFA500'
    else:
        percentage = random.uniform(85, 95)
        color = '#4CAF50'

    chart_data = {
        'percentage': round(percentage),
        'color': color,
        'remaining_percentage': round(100 - percentage)
    }

    logging.debug(f"Epistemic chart data for task_id {task_id}, uncertainty_level {uncertainty_level}, percentage {percentage}: {json.dumps(chart_data)}")
    return chart_data, round(percentage)

def aleatoric_charts(level, predicted_charge, true_charge, task_id):
    np.random.seed(task_id)
    error = abs(true_charge - predicted_charge)
    if level == 1:
        max_spread = min(5 * error, 70000)
        x = np.linspace(max(0, predicted_charge - max_spread / 2), min(70000, predicted_charge + max_spread / 2), 100)
        y1 = norm.pdf(x, predicted_charge - max_spread / 4, max_spread / 6)
        y2 = norm.pdf(x, predicted_charge + max_spread / 4, max_spread / 6)
        y = 0.5 * y1 + 0.5 * y2
    elif level == 2:
        max_spread = min(3 * error, 70000)
        x = np.linspace(max(0, predicted_charge - max_spread / 2), min(70000, predicted_charge + max_spread / 2), 100)
        y1 = norm.pdf(x, predicted_charge - max_spread / 4, max_spread / 8)
        y2 = norm.pdf(x, predicted_charge + max_spread / 4, max_spread / 8)
        y = 0.5 * y1 + 0.5 * y2
    elif level == 3:
        max_spread = min(4 * error, 70000)
        x = np.linspace(max(0, predicted_charge - max_spread / 2), min(70000, predicted_charge + max_spread / 2), 100)
        y = norm.pdf(x, predicted_charge + np.random.uniform(-max_spread / 4, max_spread / 4), max_spread / 4)
    else:
        max_spread = min(2 * error, 70000)
        x = np.linspace(max(0, predicted_charge - max_spread / 2), min(70000, predicted_charge + max_spread / 2), 100)
        y = norm.pdf(x, predicted_charge + np.random.uniform(-max_spread / 8, max_spread / 8), max_spread / 8)

    y = y / y.max() * 8

    plt.figure(figsize=(5, 2.5))
    plt.plot(x, y, color='#4CAF50')
    plt.fill_between(x, y, alpha=0.3, color='#4CAF50')
    plt.axvline(predicted_charge, color='#D32F2F', linestyle='--', label='Predicted Charge')
    plt.xlabel('Insurance Charge (USD)')
    plt.ylabel('Likelihood')
    plt.title('Charge Likelihood')
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def sample_rows(condition):
    condition_map = {1: 'AI_only', 2: 'epistemic', 3: 'aleatoric'}
    condition_str = condition_map[condition]
    valid_rows = df[df['condition'].str.lower() == condition_str.lower()]
    if len(valid_rows) != 16:
        logging.warning(f"Expected 16 rows for condition {condition_str}, found {len(valid_rows)}")
        valid_rows = df[df['condition'].isin(['AI_only', 'epistemic', 'aleatoric'])].sample(16, random_state=np.random.randint(1000))
    sampled_rows = valid_rows.to_dict('records')
    random.shuffle(sampled_rows)
    for i, row in enumerate(sampled_rows):
        row['ID'] = i + 1
    return sampled_rows

def get_practice_data():
    return df[df['condition'] == 'practice'].sample(5, random_state=np.random.randint(1000)).to_dict('records')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    available_conditions = [c for c, count in PARTICIPANT_COUNTS.items() if count < MAX_PER_CONDITION]
    if not available_conditions:
        return "Experiment is full!", 403

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
    session['tasks'] = sample_rows(condition)
    session['task_index'] = 0
    session['responses'] = []

    return redirect(url_for('practice'))

@app.route('/practice', methods=['GET', 'POST'])
def practice():
    practice_index = session.get('practice_index', 0)
    practice_data = get_practice_data()
    if practice_index >= 5:
        return redirect(url_for('transition'))

    practice_row = practice_data[practice_index]
    customer_number = practice_index + 1

    if request.method == 'POST':
        initial_guess = request.form.get('initial_guess_value')
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 70000:
                raise ValueError
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid charge guess. Please select a value between 1 and 70,000 USD.", 400

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
        'smoker': practice_row['smoker'],
        'region': practice_row['region']
    }
    return render_template('practice.html',
                         customer_info=customer_info,
                         customer_number=customer_number,
                         averages=AVERAGES)

@app.route('/transition')
def transition():
    return render_template('transition.html')

@app.route('/task', methods=['GET', 'POST'])
def task():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    if task_index >= 16:
        max_retries = 3
        conn = None
        for attempt in range(max_retries):
            conn = db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    for response in session.get('responses', []):
                        task_data = next(t for t in tasks if t['ID'] == response['ID'])
                        uncertainty_level = task_data.get('uncertainty_level', 1)
                        try:
                            uncertainty_level = int(uncertainty_level)
                            if uncertainty_level < 1 or uncertainty_level > 4:
                                logging.error(f"Invalid uncertainty_level: {uncertainty_level}")
                                uncertainty_level = 1
                        except (ValueError, TypeError):
                            logging.error(f"Non-integer uncertainty_level: {uncertainty_level}")
                            uncertainty_level = 1
                        cur.execute(
                            "INSERT INTO responses (participant_id, task_number, condition, initial_guess, final_guess, predicted_charge, uncertainty_level, "
                            "true_charge, age, sex, bmi, children, smoker, region, prediction_error, total_uncertainty_std, epistemic_uncertainty_std, aleatoric_uncertainty_std, task_duration_ms, created_at) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (session['participant_id'], response['Task_Number'], response['Condition'], response['Initial_Guess'],
                             response['Final_Guess'], response['Predicted_Charge'], uncertainty_level,
                             task_data['true_charges'], task_data['age'], task_data['sex'], task_data['bmi'],
                             task_data['children'], task_data['smoker'], task_data['region'], task_data['prediction_error'],
                             task_data['total_uncertainty_std'], task_data['epistemic_uncertainty_std'], task_data['aleatoric_uncertainty_std'],
                             response['Task_Duration_ms'], response['Created_At'])
                        )
                    conn.commit()
                logging.info("Database commit successful")
                break
            except psycopg2.OperationalError as e:
                logging.error(f"Database OperationalError on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                conn.close()
                db_pool.putconn(conn, close=True)
                return "Database connection failed after retries", 500
            except Exception as e:
                logging.error(f"Database error: {str(e)}")
                conn.rollback()
                db_pool.putconn(conn, close=True)
                return f"Error saving data: {str(e)}", 500
            finally:
                if conn and not conn.closed:
                    db_pool.putconn(conn)
        session.pop('responses', None)
        session.pop('tasks', None)
        session.pop('task_index', None)
        return render_template('end.html', participant_id=session.get('participant_id'))

    task_data = tasks[task_index]
    condition = session['condition']
    customer_number = task_index + 1

    if request.method == 'POST':
        initial_guess = request.form.get('initial_guess_value')
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 70000:
                raise ValueError
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid charge guess. Please select a value between 1 and 70,000 USD.", 400
        session['current_initial_guess'] = initial_guess
        session['task_start_time'] = time.time()  # Record task start time
        return redirect(url_for('stage2'))

    customer_info = {
        'age': task_data['age'],
        'sex': task_data['sex'],
        'bmi': task_data['bmi'],
        'smoker': task_data['smoker'],
        'region': task_data['region']
    }
    session['task_start_time'] = time.time()  # Record task start time for GET request
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
        action = request.form.get('action')
        if action == 'view' or not session.get('show_ai_info', False):
            session['show_ai_info'] = True
            return redirect(url_for('stage2'))
        elif action == 'submit':
            final_guess = request.form.get('final_guess_value')
            try:
                final_guess = float(final_guess)
                if final_guess < 1 or final_guess > 70000:
                    raise ValueError
            except (ValueError, TypeError):
                logging.error(f"Invalid final_guess: {final_guess}")
                return "Invalid charge guess. Please select a value between 1 and 70,000 USD.", 400
            session['current_final_guess'] = final_guess
            return redirect(url_for('stage3'))

    predicted_charge = task_data['predicted_charges']
    info_data = None
    epistemic_percentage = None
    if condition == 2:
        info_data, epistemic_percentage = epistemic_charts(task_data['uncertainty_level'], task_data['ID'])
    elif condition == 3:
        info_data = aleatoric_charts(task_data['uncertainty_level'], predicted_charge, task_data['true_charges'], task_data['ID'])

    customer_info = {
        'age': task_data['age'],
        'sex': task_data['sex'],
        'bmi': task_data['bmi'],
        'smoker': task_data['smoker'],
        'region': task_data['region']
    }

    return render_template(f'review_condition{condition}.html',
                         predicted_charge=predicted_charge,
                         initial_guess=initial_guess,
                         show_ai_info=session.get('show_ai_info', False),
                         info_data=info_data,
                         customer_number=customer_number,
                         customer_info=customer_info,
                         epistemic_percentage=epistemic_percentage)

@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    if task_index >= 16:
        logging.error(f"Invalid task_index: {task_index}")
        return redirect(url_for('task'))

    task_data = tasks[task_index]
    initial_guess = session.get('current_initial_guess')
    final_guess = session.get('current_final_guess')
    task_start_time = session.get('task_start_time', time.time())
    task_duration_ms = (time.time() - task_start_time) * 1000  # Duration in milliseconds
    created_at = datetime.datetime.now()  # Record task completion timestamp

    if request.method == 'POST':
        session['responses'].append({
            'ID': task_data['ID'],
            'Task_Number': task_index + 1,
            'Condition': session['condition'],
            'Initial_Guess': initial_guess,
            'Final_Guess': final_guess,
            'Predicted_Charge': task_data['predicted_charges'],
            'Uncertainty_Level': task_data['uncertainty_level'],
            'True_Charge': task_data['true_charges'],
            'Task_Duration_ms': task_duration_ms,
            'Created_At': created_at
        })
        session['task_index'] += 1
        session.pop('show_ai_info', None)
        session.pop('current_initial_guess', None)
        session.pop('current_final_guess', None)
        session.pop('task_start_time', None)
        return redirect(url_for('task'))

    return render_template('results.html',
                         customer_number=task_index + 1,
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