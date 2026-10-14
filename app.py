import os
from flask import Flask, jsonify, request, render_template
from jobdata import load_data, make_dashboard_data, Recommender

DATA_PATH = os.environ.get('DATA_PATH', '../data/upwork_job.csv')

app = Flask(__name__)
state = {}


def load_everything():
    df = load_data(DATA_PATH)
    state['recommender'] = Recommender(df)
    state['dashboard'] = make_dashboard_data(df)
    print('data loaded, jobs:', len(df))


load_everything()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'jobs_in_data': state['dashboard']['total_jobs']})


@app.route('/api/countries')
def countries():
    return jsonify(state['recommender'].countries)


@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend():
    # works with JSON body (POST) or query string (GET)
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
    else:
        data = request.args

    skills = (data.get('skills') or '').strip()
    if not skills:
        return jsonify({'error': 'skills is required'}), 400

    job_type = data.get('job_type', 'any')
    if job_type not in ('any', 'hourly', 'fixed'):
        return jsonify({'error': "job_type must be 'any', 'hourly' or 'fixed'"}), 400

    try:
        min_hourly = float(data.get('min_hourly') or 0)
        top_n = int(data.get('top_n') or 10)
    except ValueError:
        return jsonify({'error': 'min_hourly and top_n must be numbers'}), 400
    top_n = max(1, min(top_n, 50))

    country = data.get('country') or None
    result = state['recommender'].recommend(skills, job_type, min_hourly, country, top_n)
    return jsonify({'query': skills, 'count': len(result), 'results': result})


@app.route('/api/dashboard')
def dashboard_data():
    return jsonify(state['dashboard'])


@app.route('/api/refresh', methods=['POST'])
def refresh():
    # call this after a new month of data is added to the csv file
    load_everything()
    return jsonify({'status': 'reloaded', 'jobs_in_data': state['dashboard']['total_jobs']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
