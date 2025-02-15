from sc import WorkoutRecommender
import numpy as np
import json
import os

def generate_visualization_data():
    recommender = WorkoutRecommender()
    
    # Create a finer grid for smoother visualization
    bmi_range = np.linspace(15, 40, 50)
    hr_range = np.linspace(40, 200, 50)
    age = 30
    
    # Create meshgrid for 3D surface
    BMI, HR = np.meshgrid(bmi_range, hr_range)
    intensity = np.zeros_like(BMI)
    calories = np.zeros_like(BMI)
    rest = np.zeros_like(BMI)
    workout_types = np.zeros_like(BMI)
    
    # Calculate values for each point
    for i in range(len(hr_range)):
        for j in range(len(bmi_range)):
            result = recommender.recommend(BMI[i,j], HR[i,j], age)
            if result:
                intensity[i,j] = result['workout_intensity']
                calories[i,j] = result['caloric_burn']
                rest[i,j] = result['rest_period']
                workout_map = {'Yoga': 1, 'Cardio': 2, 'Strength Training': 3, 'HIIT': 4}
                workout_types[i,j] = workout_map.get(result['workout_type'], 0)
    
    # Save visualization data
    visualization_data = {
        'bmi': BMI.tolist(),
        'heart_rate': HR.tolist(),
        'intensity': intensity.tolist(),
        'calories': calories.tolist(),
        'rest': rest.tolist(),
        'workout_types': workout_types.tolist()
    }
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join('static', 'data')
    os.makedirs(static_dir, exist_ok=True)
    
    with open(os.path.join(static_dir, 'visualization_data.json'), 'w') as f:
        json.dump(visualization_data, f)

if __name__ == "__main__":
    generate_visualization_data() 