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

def confidence_interval_chart(level, predicted_charge, task_id):
    np.random.seed(task_id)
    if level == 1:  # Highest uncertainty
        percentage = 0.5  # ±50%
    elif level == 2:
        percentage = 0.35  # ±35%
    elif level == 3:
        percentage = 0.2  # ±20%
    else:  # Lowest uncertainty
        percentage = 0.07  # ±7%

    lower_bound = max(0, predicted_charge * (1 - percentage))
    upper_bound = min(70000, predicted_charge * (1 + percentage))

    fig = plt.figure(figsize=(5, 1), facecolor='none', frameon=False)
    ax = plt.gca()
    ax.hlines(y=0, xmin=lower_bound, xmax=upper_bound, colors='#4CAF50', linewidth=5)
    ax.plot([lower_bound, lower_bound], [-0.2, 0.2], color='#4CAF50', linewidth=2)  # Left vertical dash
    ax.plot([upper_bound, upper_bound], [-0.2, 0.2], color='#4CAF50', linewidth=2)  # Right vertical dash
    ax.plot(predicted_charge, 0, 'ro', markersize=10)  # Red dot in middle
    ax.set_xlim(lower_bound - 1000, upper_bound + 1000)
    ax.set_ylim(-0.5, 0.5)
    ax.text(lower_bound, -0.4, f'${int(lower_bound)}', ha='center', va='top', fontsize=10)
    ax.text(upper_bound, -0.4, f'${int(upper_bound)}', ha='center', va='top', fontsize=10)
    ax.set_facecolor('#f9fafb')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True, dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

def sample_rows(condition):
    condition_map = {1: 'AI_only', 2: 'epistemic', 3: 'aleatoric'}
    condition_str = condition_map[condition]
    if condition_str == 'AI_only':
        valid_rows = df[df['condition'].str.lower() == condition_str.lower()].head(8)
    else:
        valid_rows = pd.concat([
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 1)].head(2),
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 2)].head(2),
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 3)].head(2),
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 4)].head(2)
        ])
    if len(valid_rows) != 8:
        logging.warning(f"Expected 8 rows for condition {condition_str}, found {len(valid_rows)}")
        valid_rows = df[df['condition'].isin(['AI_only', 'epistemic', 'aleatoric'])].head(8)
    sampled_rows = valid_rows.to_dict('records')
    for i, row in enumerate(sampled_rows):
        row['ID'] = i + 1
    return sampled_rows

def get_practice_data():
    return df[df['condition'] == 'practice'].head(5).to_dict('records')

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
    session['performance_wins'] = 0  # Track wins for main tasks

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
            return "Invalid cost estimate. Please select a value between 1 and 70,000 USD.", 400

        user_error = abs(initial_guess - practice_row['true_charges'])
        ai_error = abs(practice_row['predicted_charges'] - practice_row['true_charges'])
        performance_message = "Your estimate was closer to the true medical cost than the AI’s prediction" if user_error <= ai_error else "The AI’s prediction was closer to the true medical cost than your estimate"

        session['practice_index'] += 1
        return render_template('practice_result.html',
                             customer_number=customer_number,
                             customer_info=practice_row,
                             initial_guess=initial_guess,
                             true_charge=practice_row['true_charges'],
                             ai_prediction=practice_row['predicted_charges'],
                             performance_message=performance_message)

    customer_info = {
        'age': practice_row['age'],
        'sex': practice_row['sex'],
        'bmi': practice_row['bmi'],
        'smoker': practice_row['smoker']
    }
    return render_template('practice.html',
                         customer_info=customer_info,
                         customer_number=customer_number)

@app.route('/transition')
def transition():
    return render_template('transition.html')

@app.route('/task', methods=['GET', 'POST'])
def task():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    logging.debug(f"Task route: task_index={task_index}, tasks_length={len(tasks)}")
    if task_index >= 8:
        max_retries = 3
        conn = None
        performance_wins = session.get('performance_wins', 0)
        performance_score = 20 + (performance_wins * 10)  # 0 wins = 20%, 1 win = 30%, ..., 8 wins = 100%
        bonus = (performance_score / 100) * 2.0
        logging.info(f"Participant {session['participant_id']} performance: {performance_score}%, bonus: ${bonus:.2f}")
        for attempt in range(max_retries):
            conn = db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'responses'")
                    schema = [row[0] for row in cur.fetchall()]
                    logging.debug(f"Responses table schema: {schema}")
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
                        log_response = response.copy()
                        log_response['Created_At'] = log_response['Created_At'].isoformat()
                        logging.debug(f"Response data: {json.dumps(log_response)}")
                        insert_args = (
                            session['participant_id'], response['Task_Number'], response['Condition'], response['Initial_Guess'],
                            response['Final_Guess'], response['Predicted_Charge'], uncertainty_level,
                            task_data['true_charges'], task_data['age'], task_data['sex'], task_data['bmi'],
                            task_data['children'], task_data['smoker'], task_data['prediction_error'],
                            task_data['total_uncertainty_std'], task_data['epistemic_uncertainty_std'], task_data['aleatoric_uncertainty_std'],
                            response['Task_Duration_ms'], response['Created_At']
                        )
                        logging.debug(f"INSERT arguments: {insert_args}")
                        cur.execute(
                            "INSERT INTO responses (participant_id, task_number, condition, initial_guess, final_guess, predicted_charge, uncertainty_level, "
                            "true_charge, age, sex, bmi, children, smoker, prediction_error, total_uncertainty_std, epistemic_uncertainty_std, aleatoric_uncertainty_std, task_duration_ms, created_at) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            insert_args
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
        return render_template('end.html', participant_id=session.get('participant_id'), performance_score=performance_score)

    task_data = tasks[task_index]
    condition = session['condition']
    customer_number = task_index + 1

    if request.method == 'POST':
        logging.debug(f"Task form data received: {request.form}")
        initial_guess = request.form.get('initial_guess_value')
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 70000:
                raise ValueError("Initial guess out of range")
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid cost estimate. Please select a value between 1 and 70,000 USD.", 400
        session['current_initial_guess'] = initial_guess
        session['task_start_time'] = time.time()
        return redirect(url_for('stage2'))

    customer_info = {
        'age': task_data['age'],
        'sex': task_data['sex'],
        'bmi': task_data['bmi'],
        'smoker': task_data['smoker']
    }
    return render_template('assessment.html',
                         customer_info=customer_info,
                         customer_number=customer_number)

@app.route('/stage2', methods=['GET', 'POST'])
def stage2():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    logging.debug(f"Stage2 route: task_index={task_index}, tasks_length={len(tasks)}")
    if task_index >= len(tasks):
        logging.error(f"Invalid task_index: {task_index}, tasks_length={len(tasks)}")
        return "Invalid task index", 500
    task_data = tasks[task_index]
    condition = session['condition']
    initial_guess = session.get('current_initial_guess')
    customer_number = task_index + 1
    performance_message = None
    submitted = False
    final_guess = session.get('current_final_guess', initial_guess)

    if request.method == 'POST':
        logging.debug(f"Stage2 form data received: {request.form}")
        final_guess = request.form.get('final_guess_value')
        logging.debug(f"Stage2 POST: final_guess={final_guess}, initial_guess={initial_guess}")
        try:
            final_guess = float(final_guess)
            if final_guess < 1 or final_guess > 70000:
                raise ValueError("Final guess out of range")
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid final_guess: {final_guess}, Error: {str(e)}")
            return "Invalid cost estimate. Please select a value between 1 and 70,000 USD.", 400

        user_error = abs(final_guess - task_data['true_charges'])
        ai_error = abs(task_data['predicted_charges'] - task_data['true_charges'])
        performance_message = "Your estimate was closer to the true medical cost than the AI’s prediction" if user_error <= ai_error else "The AI’s prediction was closer to the true medical cost than your estimate"
        if user_error <= ai_error:
            session['performance_wins'] = session.get('performance_wins', 0) + 1

        session['responses'].append({
            'ID': task_data['ID'],
            'Task_Number': task_index + 1,
            'Condition': session['condition'],
            'Initial_Guess': initial_guess,
            'Final_Guess': final_guess,
            'Predicted_Charge': task_data['predicted_charges'],
            'Uncertainty_Level': task_data['uncertainty_level'],
            'True_Charge': task_data['true_charges'],
            'Task_Duration_ms': (time.time() - session.get('task_start_time', time.time())) * 1000,
            'Created_At': datetime.datetime.now(),
            'Performance_Message': performance_message
        })
        session['current_final_guess'] = final_guess
        submitted = True

        predicted_charge = task_data['predicted_charges']
        info_data = None
        epistemic_percentage = None
        if condition == 2:
            info_data, epistemic_percentage = epistemic_charts(task_data['uncertainty_level'], task_data['ID'])
        elif condition == 3:
            info_data = confidence_interval_chart(task_data['uncertainty_level'], predicted_charge, task_data['ID'])

        customer_info = {
            'age': task_data['age'],
            'sex': task_data['sex'],
            'bmi': task_data['bmi'],
            'smoker': task_data['smoker']
        }

        return render_template(f'review_condition{condition}.html',
                             predicted_charge=predicted_charge,
                             initial_guess=initial_guess,
                             final_guess=final_guess,
                             info_data=info_data,
                             customer_number=customer_number,
                             customer_info=customer_info,
                             epistemic_percentage=epistemic_percentage,
                             performance_message=performance_message,
                             submitted=submitted)

    logging.debug(f"Stage2 GET: final_guess={final_guess}, initial_guess={initial_guess}")
    predicted_charge = task_data['predicted_charges']
    info_data = None
    epistemic_percentage = None
    if condition == 2:
        info_data, epistemic_percentage = epistemic_charts(task_data['uncertainty_level'], task_data['ID'])
    elif condition == 3:
        info_data = confidence_interval_chart(task_data['uncertainty_level'], predicted_charge, task_data['ID'])

    customer_info = {
        'age': task_data['age'],
        'sex': task_data['sex'],
        'bmi': task_data['bmi'],
        'smoker': task_data['smoker']
    }

    return render_template(f'review_condition{condition}.html',
                         predicted_charge=predicted_charge,
                         initial_guess=initial_guess,
                         final_guess=final_guess,
                         info_data=info_data,
                         customer_number=customer_number,
                         customer_info=customer_info,
                         epistemic_percentage=epistemic_percentage,
                         performance_message=performance_message,
                         submitted=submitted)

@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    logging.debug(f"Stage3 route: task_index={task_index}, tasks_length={len(tasks)}")
    if task_index >= 8:
        return redirect(url_for('task'))

    task_data = tasks[task_index]
    initial_guess = session.get('current_initial_guess')
    final_guess = session.get('current_final_guess')
    performance_message = session['responses'][-1]['Performance_Message'] if session['responses'] else None

    if request.method == 'POST':
        session['task_index'] = task_index + 1
        logging.debug(f"Incremented task_index to {session['task_index']}")
        session.pop('current_initial_guess', None)
        session.pop('current_final_guess', None)
        session.pop('task_start_time', None)
        return redirect(url_for('task'))

    return render_template('results.html',
                         customer_number=task_index + 1,
                         initial_guess=initial_guess,
                         final_guess=final_guess,
                         ai_prediction=task_data['predicted_charges'],
                         performance_message=performance_message)

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