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

CSV_URL = "https://drive.google.com/uc?id=1eFFZZW0mYSmQovl7vRkvZpV_2UE8XCv3"
df = pd.read_csv(CSV_URL)

# Log initial data state
logging.debug(f"Initial DataFrame shape: {df.shape}")
logging.debug(f"Initial 'age' column values: {df['age'].tolist()[:14]}")
logging.debug(f"Initial 'bmi' column values: {df['bmi'].tolist()[:14]}")
logging.debug(f"Initial 'children' column values: {df['children'].tolist()[:14]}")

# Handle non-numeric age and bmi for practice records (rows 7-14, indices 6-13)
def parse_age(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        if isinstance(value, str):
            # Remove parentheses for ranges like "(20-35)"
            value = value.strip('()')
            if '-' in value:
                try:
                    low, high = map(float, value.split('-'))
                    return int((low + high) / 2)
                except ValueError as e:
                    logging.error(f"Failed to parse age range '{value}': {str(e)}")
                    return np.nan
            elif 'below' in value.lower():
                try:
                    return int(float(value.split()[-1]) * 0.833)  # e.g., 'below 30' -> 25
                except ValueError as e:
                    logging.error(f"Failed to parse age '{value}': {str(e)}")
                    return np.nan
            elif 'above' in value.lower():
                try:
                    return int(float(value.split()[-1]) * 1.2)
                except ValueError as e:
                    logging.error(f"Failed to parse age '{value}': {str(e)}")
                    return np.nan
        logging.error(f"Invalid age value: {value}")
        return np.nan

def parse_bmi(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        if isinstance(value, str):
            # Remove parentheses for ranges like "(20-30)"
            value = value.strip('()')
            if '-' in value:
                try:
                    low, high = map(float, value.split('-'))
                    return (low + high) / 2
                except ValueError as e:
                    logging.error(f"Failed to parse bmi range '{value}': {str(e)}")
                    return np.nan
            elif 'below' in value.lower():
                try:
                    return float(value.split()[-1]) * 0.833
                except ValueError as e:
                    logging.error(f"Failed to parse bmi '{value}': {str(e)}")
                    return np.nan
            elif 'above' in value.lower():
                try:
                    return float(value.split()[-1]) * 1.2
                except ValueError as e:
                    logging.error(f"Failed to parse bmi '{value}': {str(e)}")
                    return np.nan
        logging.error(f"Invalid bmi value: {value}")
        return np.nan

# Create a copy of the DataFrame for example display (rows 7-14 keep strings)
df_examples = df.copy()

# Process age and bmi for main and practice tasks (rows 1-6, and non-practice)
df['age'] = df['age'].apply(parse_age)
df['bmi'] = df['bmi'].apply(parse_bmi)

# Handle missing or invalid values
for col in ['age', 'children']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isna().any():
        logging.warning(f"Found NaN values in column {col}: {df[col].isna().sum()} NaNs")
        if col == 'age':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    df[col] = df[col].astype(int)
    logging.debug(f"Processed '{col}' column: {df[col].tolist()[:14]}")

for col in ['bmi', 'true_charges', 'predicted_charges', 'prediction_error', 'total_uncertainty_std', 'epistemic_uncertainty_std', 'aleatoric_uncertainty_std']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isna().any():
        logging.warning(f"Found NaN values in column {col}: {df[col].isna().sum()} NaNs")
        df[col] = df[col].fillna(0)
    df[col] = df[col].astype(float)
    logging.debug(f"Processed '{col}' column: {df[col].tolist()[:14]}")

df['smoker_enc'] = df['smoker'].map({'yes': 1, 'no': 0})
region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
df['region_enc'] = df['region'].map(region_map)
df['uncertainty_level'] = df.apply(
    lambda x: 1 if pd.isna(x['uncertainty_level']) or x['uncertainty_level'] == 'NA' or x['condition'] in ['AI_only', 'practice']
    else x['uncertainty_level'], axis=1
)
df['uncertainty_level'] = df['uncertainty_level'].astype(int)

AVERAGES = {
    'age': 35,
    'bmi': 30,
    'charges': 13270
}

PARTICIPANT_COUNTS = {1: 0, 2: 0, 3: 0}
MAX_PER_CONDITION = 200
PARTICIPANT_ORDER = []

def epistemic_charts(uncertainty_level, task_id):
    random.seed(task_id)
    if uncertainty_level == 1:
        instance_count = 1
        color = '#FF6B6B'
        reliability = 'low'
    elif uncertainty_level == 2:
        instance_count = 5
        color = '#FFA500'
        reliability = 'medium'
    else:
        instance_count = 10
        color = '#4CAF50'
        reliability = 'satisfying'

    chart_data = {
        'instance_count': instance_count,
        'color': color,
        'remaining_instances': 10 - instance_count,
        'label_position': 'left' if instance_count == 1 else 'center',
        'reliability': reliability
    }

    logging.debug(f"Epistemic chart data for task_id {task_id}, uncertainty_level {uncertainty_level}, instance_count {instance_count}: {json.dumps(chart_data)}")
    return chart_data, instance_count

def confidence_interval_chart(level, predicted_charge, task_id):
    np.random.seed(task_id)
    if level == 1:
        percentage = 0.6  # ±60%
    elif level == 2:
        percentage = 0.3  # ±30%
    else:
        percentage = 0.05  # ±5%

    lower_bound = max(0, predicted_charge * (1 - percentage))
    upper_bound = min(30000, predicted_charge * (1 + percentage))

    fig = plt.figure(figsize=(5, 1), facecolor='none', frameon=False)
    ax = plt.gca()
    ax.hlines(y=0, xmin=lower_bound, xmax=upper_bound, colors='#4CAF50', linewidth=5)
    ax.plot([lower_bound, lower_bound], [-0.2, 0.2], color='#4CAF50', linewidth=2)
    ax.plot([upper_bound, upper_bound], [-0.2, 0.2], color='#4CAF50', linewidth=2)
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
        valid_rows = df[df['condition'].str.lower() == condition_str.lower()].sample(n=6, random_state=random.randint(1, 10000))
    else:
        valid_rows = pd.concat([
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 1)].sample(n=2, random_state=random.randint(1, 10000)),
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 2)].sample(n=2, random_state=random.randint(1, 10000)),
            df[(df['condition'].str.lower() == condition_str.lower()) & (df['uncertainty_level'] == 3)].sample(n=2, random_state=random.randint(1, 10000))
        ])
    if len(valid_rows) != 6:
        logging.warning(f"Expected 6 rows for condition {condition_str}, found {len(valid_rows)}")
        valid_rows = df[df['condition'].isin(['AI_only', 'epistemic', 'aleatoric'])].sample(n=6, random_state=random.randint(1, 10000))
    sampled_rows = valid_rows.to_dict('records')
    random.shuffle(sampled_rows)
    for i, row in enumerate(sampled_rows):
        row['ID'] = i + 1
    return sampled_rows

def get_practice_data():
    return df[df['condition'] == 'practice'].head(6).to_dict('records')

def get_example_data():
    practice_df = df_examples[df_examples['condition'] == 'practice'][6:14]
    non_smokers = practice_df[practice_df['smoker'] == 'no'].head(4).to_dict('records')
    smokers = practice_df[practice_df['smoker'] == 'yes'].head(4).to_dict('records')
    return {'non_smokers': non_smokers, 'smokers': smokers}

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

    prolific_id = request.form.get('prolific_id', '').strip()
    if not prolific_id:
        return "Prolific ID is required.", 400

    participant_id = str(uuid.uuid4())

    PARTICIPANT_COUNTS[condition] += 1
    PARTICIPANT_ORDER.append(condition)
    logging.info(f"Assigned participant with Prolific ID {prolific_id} to condition {condition}")

    session['condition'] = condition
    session['prolific_id'] = prolific_id
    session['participant_id'] = participant_id
    session['practice_index'] = 0
    session['tasks'] = sample_rows(condition)
    session['task_index'] = 0
    session['responses'] = []
    session['performance_wins'] = 0
    session['attention_errors'] = 0

    return redirect(url_for('examples'))

@app.route('/examples')
def examples():
    example_data = get_example_data()
    return render_template('examples.html', example_data=example_data)

@app.route('/practice', methods=['GET', 'POST'])
def practice():
    practice_index = session.get('practice_index', 0)
    practice_data = get_practice_data()
    if practice_index >= 6:
        return redirect(url_for('transition'))

    practice_row = practice_data[practice_index]
    customer_number = practice_index + 1

    if request.method == 'POST':
        initial_guess = request.form.get('initial_guess_value')
        logging.debug(f"Practice POST: initial_guess={initial_guess}, form_data={request.form.to_dict()}")
        if not initial_guess:
            return "Estimated Medical Cost is required.", 400
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 30000:
                raise ValueError
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid cost estimate. Please select a value between 1 and 30,000 USD.", 400

        user_error = abs(initial_guess - practice_row['true_charges'])
        automated_algorithm_error = abs(practice_row['predicted_charges'] - practice_row['true_charges'])
        performance_message = "Your estimate was closer to the true medical cost than the automated algorithm’s prediction" if user_error <= automated_algorithm_error else "The automated algorithm’s prediction was closer to the true medical cost than your estimate"

        session['practice_index'] = session.get('practice_index', 0) + 1
        return render_template('practice_result.html',
                             customer_number=customer_number,
                             customer_info=practice_row,
                             initial_guess=initial_guess,
                             true_charge=practice_row['true_charges'],
                             automated_algorithm_prediction=practice_row['predicted_charges'],
                             performance_message=performance_message)

    customer_info = {
        'age': practice_row['age'],
        'bmi': practice_row['bmi'],
        'smoker': practice_row['smoker']
    }
    return render_template('practice.html',
                         customer_info=customer_info,
                         customer_number=customer_number)

@app.route('/transition')
def transition():
    return render_template('transition.html')

@app.route('/attention_check', methods=['GET', 'POST'])
def attention_check():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    if task_index not in [2, 5]:
        return redirect(url_for('task'))

    if request.method == 'POST':
        answer = request.form.get('answer')
        if task_index == 2:
            correct_answer = 'yes' if tasks[1]['smoker'] == 'yes' else 'no'
            question = 'smoker'
        else:  # task_index == 5
            correct_answer = 'yes' if tasks[4]['age'] < 35 else 'no'
            question = 'age_below_35'
        if answer != correct_answer:
            session['attention_errors'] = session.get('attention_errors', 0) + 1
        logging.debug(f"Attention check: task_index={task_index}, question={question}, answer={answer}, correct={correct_answer}, errors={session['attention_errors']}")
        if task_index == 5 and session['attention_errors'] >= 2:
            return redirect(url_for('end_with_error'))
        return redirect(url_for('task'))

    if task_index == 2:
        question = "Was Patient 2 a smoker?"
    else:  # task_index == 5
        question = "Was Patient 5 below 35 years old?"
    return render_template('attention_check.html', question=question, customer_number=task_index + 1)

@app.route('/task', methods=['GET', 'POST'])
def task():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    logging.debug(f"Task route: task_index={task_index}, tasks_length={len(tasks)}")
    if task_index >= 6:
        max_retries = 3
        conn = None
        performance_wins = session.get('performance_wins', 0)
        performance_score = 20 + (performance_wins * 13.33)  # Adjusted for 6 tasks: 0 wins = 20%, 1 win = 33.33%, ..., 6 wins = 100%
        bonus = (performance_score / 100) * 2.0
        attention_errors = session.get('attention_errors', 0)
        logging.info(f"Participant with Prolific ID {session.get('prolific_id')} performance: {performance_score}%, bonus: ${bonus:.2f}, attention_errors: {attention_errors}")
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
                            if uncertainty_level < 1 or uncertainty_level > 3:
                                logging.error(f"Invalid uncertainty_level: {uncertainty_level}")
                                uncertainty_level = 1
                        except (ValueError, TypeError):
                            logging.error(f"Non-integer uncertainty_level: {uncertainty_level}")
                            uncertainty_level = 1
                        log_response = response.copy()
                        log_response['Created_At'] = log_response['Created_At'].isoformat()
                        logging.debug(f"Response data: {json.dumps(log_response)}")
                        insert_args = (
                            session['prolific_id'], response['Task_Number'], response['Condition'], response['Initial_Guess'],
                            response['Final_Guess'], response['Predicted_Charge'], uncertainty_level,
                            task_data['true_charges'], task_data['age'], task_data['bmi'],
                            task_data['children'], task_data['smoker'], task_data['prediction_error'],
                            task_data['total_uncertainty_std'], task_data['epistemic_uncertainty_std'], task_data['aleatoric_uncertainty_std'],
                            response['Task_Duration_ms'], response['Created_At'], attention_errors
                        )
                        logging.debug(f"INSERT arguments: {insert_args}")
                        cur.execute(
                            "INSERT INTO responses (prolific_id, task_number, condition, initial_guess, final_guess, predicted_charge, uncertainty_level, "
                            "true_charge, age, bmi, children, smoker, prediction_error, total_uncertainty_std, epistemic_uncertainty_std, aleatoric_uncertainty_std, task_duration_ms, created_at, attention_errors) "
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
        session.pop('attention_errors', None)
        return render_template('end.html', performance_score=performance_score)

    task_data = tasks[task_index]
    condition = session['condition']
    customer_number = task_index + 1

    if request.method == 'POST':
        initial_guess = request.form.get('initial_guess_value')
        logging.debug(f"Task POST: initial_guess={initial_guess}, form_data={request.form.to_dict()}")
        if not initial_guess:
            return "Estimated Medical Cost is required.", 400
        try:
            initial_guess = float(initial_guess)
            if initial_guess < 1 or initial_guess > 30000:
                raise ValueError
        except (ValueError, TypeError):
            logging.error(f"Invalid initial_guess: {initial_guess}")
            return "Invalid cost estimate. Please select a value between 1 and 30,000 USD.", 400
        session['current_initial_guess'] = initial_guess
        session['task_start_time'] = time.time()
        return redirect(url_for('stage2'))

    customer_info = {
        'age': task_data['age'],
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
        logging.debug(f"Stage2 form data received: {request.form.to_dict()}")
        final_guess = request.form.get('final_guess_number') or request.form.get('final_guess_value')
        logging.debug(f"Stage2 POST: final_guess={final_guess}, initial_guess={initial_guess}")
        if not final_guess:
            return "Estimated Medical Cost is required.", 400
        try:
            final_guess = float(final_guess)
            if final_guess < 1 or final_guess > 30000:
                raise ValueError
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid final_guess: {final_guess}, Error: {str(e)}")
            return "Invalid cost estimate. Please select a value between 1 and 30,000 USD.", 400

        if final_guess == initial_guess:
            logging.warning(f"Final guess ({final_guess}) equals initial guess ({initial_guess}) for task {task_index + 1}")

        user_error = abs(final_guess - task_data['true_charges'])
        automated_algorithm_error = abs(task_data['predicted_charges'] - task_data['true_charges'])
        performance_message = "Your estimate was closer to the true medical cost than the automated algorithm’s prediction" if user_error <= automated_algorithm_error else "The automated algorithm’s prediction was closer to the true medical cost than your estimate"
        if user_error <= automated_algorithm_error:
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
        epistemic_instance_count = None
        if condition == 2:
            info_data, epistemic_instance_count = epistemic_charts(task_data['uncertainty_level'], task_data['ID'])
        elif condition == 3:
            info_data = confidence_interval_chart(task_data['uncertainty_level'], predicted_charge, task_data['ID'])

        customer_info = {
            'age': task_data['age'],
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
                             epistemic_instance_count=epistemic_instance_count,
                             performance_message=performance_message,
                             submitted=submitted,
                             task_data=task_data)

    logging.debug(f"Stage2 GET: final_guess={final_guess}, initial_guess={initial_guess}")
    predicted_charge = task_data['predicted_charges']
    info_data = None
    epistemic_instance_count = None
    if condition == 2:
        info_data, epistemic_instance_count = epistemic_charts(task_data['uncertainty_level'], task_data['ID'])
    elif condition == 3:
        info_data = confidence_interval_chart(task_data['uncertainty_level'], predicted_charge, task_data['ID'])

    customer_info = {
        'age': task_data['age'],
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
                         epistemic_instance_count=epistemic_instance_count,
                         performance_message=performance_message,
                         submitted=submitted,
                         task_data=task_data)

@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    logging.debug(f"Stage3 route: task_index={task_index}, tasks_length={len(tasks)}")
    if task_index >= 6:
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
        return redirect(url_for('attention_check') if task_index in [1, 4] else url_for('task'))

    return render_template('results.html',
                         customer_number=task_index + 1,
                         initial_guess=initial_guess,
                         final_guess=final_guess,
                         automated_algorithm_prediction=task_data['predicted_charges'],
                         performance_message=performance_message)

@app.route('/end_with_error')
def end_with_error():
    max_retries = 3
    conn = None
    performance_wins = session.get('performance_wins', 0)
    performance_score = 20 + (performance_wins * 13.33)  # Adjusted for 6 tasks
    bonus = (performance_score / 100) * 2.0
    attention_errors = session.get('attention_errors', 0)
    logging.info(f"Participant with Prolific ID {session.get('prolific_id')} ended early due to attention errors: {attention_errors}, performance: {performance_score}%, bonus: ${bonus:.2f}")
    for attempt in range(max_retries):
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'responses'")
                schema = [row[0] for row in cur.fetchall()]
                logging.debug(f"Responses table schema: {schema}")
                for response in session.get('responses', []):
                    task_data = next(t for t in session['tasks'] if t['ID'] == response['ID'])
                    uncertainty_level = task_data.get('uncertainty_level', 1)
                    try:
                        uncertainty_level = int(uncertainty_level)
                        if uncertainty_level < 1 or uncertainty_level > 3:
                            logging.error(f"Invalid uncertainty_level: {uncertainty_level}")
                            uncertainty_level = 1
                    except (ValueError, TypeError):
                        logging.error(f"Non-integer uncertainty_level: {uncertainty_level}")
                        uncertainty_level = 1
                    log_response = response.copy()
                    log_response['Created_At'] = log_response['Created_At'].isoformat()
                    logging.debug(f"Response data: {json.dumps(log_response)}")
                    insert_args = (
                        session['prolific_id'], response['Task_Number'], response['Condition'], response['Initial_Guess'],
                        response['Final_Guess'], response['Predicted_Charge'], uncertainty_level,
                        task_data['true_charges'], task_data['age'], task_data['bmi'],
                        task_data['children'], task_data['smoker'], task_data['prediction_error'],
                        task_data['total_uncertainty_std'], task_data['epistemic_uncertainty_std'], task_data['aleatoric_uncertainty_std'],
                        response['Task_Duration_ms'], response['Created_At'], attention_errors
                    )
                    logging.debug(f"INSERT arguments: {insert_args}")
                    cur.execute(
                        "INSERT INTO responses (prolific_id, task_number, condition, initial_guess, final_guess, predicted_charge, uncertainty_level, "
                        "true_charge, age, bmi, children, smoker, prediction_error, total_uncertainty_std, epistemic_uncertainty_std, aleatoric_uncertainty_std, task_duration_ms, created_at, attention_errors) "
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
    session.pop('attention_errors', None)
    return render_template('end.html', performance_score=performance_score, ended_early=True)

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