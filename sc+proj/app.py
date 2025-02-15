import os
import subprocess
from flask import Flask, render_template, request, jsonify
from sc import WorkoutRecommender
import numpy as np

app = Flask(__name__)
recommender = WorkoutRecommender()

# Create static/data directory and generate data
def setup_data():
    os.makedirs('static/data', exist_ok=True)
    subprocess.run(['python', 'generate_data.py'])

setup_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            bmi = float(request.form['bmi'])
            heart_rate = float(request.form['heart_rate'])
            age = float(request.form['age'])
            
            result = recommender.recommend(bmi, heart_rate, age)
            
            if isinstance(result, dict):
                return render_template('result.html', result=result)
            else:
                return render_template('index.html', error="Invalid input values. Please check your inputs.")
                
        except ValueError:
            return render_template('index.html', error="Please enter valid numerical values.")
    
    return render_template('index.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/get_visualization_data')
def get_visualization_data():
    # Create a finer grid for smoother visualization
    bmi_range = np.linspace(15, 40, 50)  # Increased points for smoother surface
    hr_range = np.linspace(40, 200, 50)  # Increased points for smoother surface
    age = 30  # Fixed age for visualization
    
    # Create meshgrid for 3D surface
    BMI, HR = np.meshgrid(bmi_range, hr_range)
    intensity = np.zeros_like(BMI)
    workout_types = np.zeros_like(BMI)
    
    # Calculate intensity and workout types for each point
    for i in range(len(hr_range)):
        for j in range(len(bmi_range)):
            result = recommender.recommend(BMI[i,j], HR[i,j], age)
            if result:
                intensity[i,j] = result['workout_intensity']
                # Convert workout type to numeric value
                workout_map = {'Yoga': 1, 'Cardio': 2, 'Strength Training': 3, 'HIIT': 4}
                workout_types[i,j] = workout_map.get(result['workout_type'], 0)
    
    return jsonify({
        'bmi': BMI.tolist(),
        'heart_rate': HR.tolist(),
        'intensity': intensity.tolist(),
        'workout_types': workout_types.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True) 