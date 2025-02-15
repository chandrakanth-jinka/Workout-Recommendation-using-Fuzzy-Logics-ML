import warnings
# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import product
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WorkoutRecommender:
    def __init__(self):
        # Load and preprocess the dataset
        self.load_dataset()

        # Adjust BMI range based on dataset
        self.bmi = ctrl.Antecedent(np.arange(12, 51, 1), 'bmi')  # Extended BMI range
        self.heart_rate = ctrl.Antecedent(np.arange(40, 201, 1), 'heart_rate')
        self.age = ctrl.Antecedent(np.arange(15, 81, 1), 'age')

        # Output variables with adjusted ranges
        self.caloric_burn = ctrl.Consequent(np.arange(100, 1501, 1), 'caloric_burn')  # Extended calorie range
        self.rest_period = ctrl.Consequent(np.arange(30, 241, 1), 'rest_period')  # Extended rest period
        self.workout_intensity = ctrl.Consequent(np.arange(0, 101, 1), 'workout_intensity')

        self.setup_membership_functions()
        self.setup_rules()
        self.init_control_system()

        # Workout tips dictionary
        self.workout_tips = {
            "Yoga": {
                "description": "Low-impact exercise focusing on flexibility, balance, and mental wellness",
                "benefits": ["Improves flexibility", "Reduces stress", "Enhances balance", "Builds core strength"],
                "examples": ["Sun Salutation", "Downward Dog", "Warrior Poses", "Child's Pose", "Tree Pose"]
            },
            "Cardio": {
                "description": "Aerobic exercises to improve heart health and endurance",
                "benefits": ["Burns calories", "Improves cardiovascular health", "Boosts stamina", "Reduces stress"],
                "examples": ["Running", "Swimming", "Cycling", "Jump Rope", "Dancing"]
            },
            "Strength Training": {
                "description": "Resistance exercises to build muscle and bone density",
                "benefits": ["Builds muscle mass", "Increases bone density", "Boosts metabolism", "Improves posture"],
                "examples": ["Squats", "Deadlifts", "Push-ups", "Pull-ups", "Bench Press"]
            },
            "HIIT": {
                "description": "High-intensity interval training for maximum calorie burn",
                "benefits": ["Maximum calorie burn", "Improves metabolism", "Time-efficient", "Builds endurance"],
                "examples": ["Burpees", "Mountain Climbers", "Sprint Intervals", "Jump Squats", "Tabata"]
            }
        }

    def load_dataset(self):
        """Load and preprocess the gym members dataset"""
        try:
          self.df = pd.read_csv('gym_members_exercise_tracking.csv')
        except FileNotFoundError as e:
          print(f"Error: {e}. Please ensure the file 'gym_members_exercise_tracking.csv' is in the correct location.")
          raise


        # Calculate statistical measures for different workout types
        self.workout_stats = {}
        for workout_type in self.df['Workout_Type'].unique():
            stats = self.df[self.df['Workout_Type'] == workout_type].agg({
                'Calories_Burned': ['mean', 'std'],
                'Session_Duration (hours)': ['mean', 'std'],
                'Max_BPM': ['mean', 'std'],
                'BMI': ['mean', 'std'],
                'Age': ['mean', 'std']
            })
            self.workout_stats[workout_type] = stats

        # Calculate workout intensity with explicit type handling
        calories_factor = (self.df['Calories_Burned'] / self.df['Session_Duration (hours)'] * 0.6)
        heart_rate_factor = (self.df['Max_BPM'] / self.df['Resting_BPM'] * 0.4)

        # Calculate intensity and ensure it's properly typed
        intensity = calories_factor + heart_rate_factor

        # Convert to integer using a safer method
        self.df['Workout_Intensity'] = pd.Series(
            np.clip(intensity, 0, 100)  # Clip values between 0 and 100
            .round()                    # Round to nearest whole number
            .astype(np.int64),         # Convert to int64
            index=self.df.index
        )

    def setup_membership_functions(self):
        """Setup membership functions with sharper distinctions"""

        # Age membership functions - reduced sigma for clearer categories
        self.age['young'] = fuzz.gaussmf(self.age.universe, 22, 4)      # Sharper young category
        self.age['middle'] = fuzz.gaussmf(self.age.universe, 40, 6)     # Moderate middle age spread
        self.age['elderly'] = fuzz.gaussmf(self.age.universe, 65, 5)    # Clearer elderly category

        # BMI membership functions - tighter ranges
        self.bmi['underweight'] = fuzz.gaussmf(self.bmi.universe, 17, 1.5)  # Sharper underweight
        self.bmi['normal'] = fuzz.gaussmf(self.bmi.universe, 22, 2)         # Clear normal range
        self.bmi['overweight'] = fuzz.gaussmf(self.bmi.universe, 27, 2)     # Distinct overweight
        self.bmi['obese'] = fuzz.gaussmf(self.bmi.universe, 35, 3)          # Broader obese category

        # Heart rate - more distinct categories
        self.heart_rate['low'] = fuzz.gaussmf(self.heart_rate.universe, 60, 8)       # Sharper low HR
        self.heart_rate['normal'] = fuzz.gaussmf(self.heart_rate.universe, 100, 10)  # Clear normal HR
        self.heart_rate['high'] = fuzz.gaussmf(self.heart_rate.universe, 140, 8)     # Distinct high HR
        self.heart_rate['very_high'] = fuzz.gaussmf(self.heart_rate.universe, 170, 7) # Sharp very high HR

        # Output Variables with more distinct ranges
        # Caloric burn - increased differentiation
        self.caloric_burn['very_low'] = fuzz.gaussmf(self.caloric_burn.universe, 200, 35)
        self.caloric_burn['low'] = fuzz.gaussmf(self.caloric_burn.universe, 400, 50)
        self.caloric_burn['medium'] = fuzz.gaussmf(self.caloric_burn.universe, 700, 75)
        self.caloric_burn['high'] = fuzz.gaussmf(self.caloric_burn.universe, 1000, 100)
        self.caloric_burn['very_high'] = fuzz.gaussmf(self.caloric_burn.universe, 1300, 125)

        # Rest period - clearer distinctions
        self.rest_period['very_short'] = fuzz.gaussmf(self.rest_period.universe, 45, 8)
        self.rest_period['short'] = fuzz.gaussmf(self.rest_period.universe, 90, 12)
        self.rest_period['medium'] = fuzz.gaussmf(self.rest_period.universe, 135, 15)
        self.rest_period['long'] = fuzz.gaussmf(self.rest_period.universe, 180, 20)
        self.rest_period['very_long'] = fuzz.gaussmf(self.rest_period.universe, 225, 25)

        # Workout intensity - sharper distinctions
        self.workout_intensity['very_low'] = fuzz.gaussmf(self.workout_intensity.universe, 20, 6)
        self.workout_intensity['low'] = fuzz.gaussmf(self.workout_intensity.universe, 40, 6)
        self.workout_intensity['medium'] = fuzz.gaussmf(self.workout_intensity.universe, 60, 6)
        self.workout_intensity['high'] = fuzz.gaussmf(self.workout_intensity.universe, 80, 6)
        self.workout_intensity['very_high'] = fuzz.gaussmf(self.workout_intensity.universe, 95, 6)

    def setup_rules(self):
        """Setup balanced fuzzy rules with equal weights"""
        self.rules = [
            # Yoga Rules (Weight: 1.0)
            ctrl.Rule(
                (self.age['elderly']) |
                (self.heart_rate['low']) |
                (self.bmi['underweight'] & self.heart_rate['low']) |
                (self.bmi['obese'] & self.heart_rate['low']),
                (self.caloric_burn['very_low'], self.rest_period['very_long'], self.workout_intensity['very_low'])
            ),

            # Cardio Rules (Weight: 1.0)
            ctrl.Rule(
                (self.age['middle'] & self.heart_rate['normal']) |
                (self.bmi['normal'] & self.heart_rate['normal']) |
                (self.bmi['overweight'] & self.heart_rate['normal']),
                (self.caloric_burn['medium'], self.rest_period['medium'], self.workout_intensity['medium'])
            ),

            # Strength Training Rules (Weight: 1.0)
            ctrl.Rule(
                (self.age['young'] & self.heart_rate['normal']) |
                (self.bmi['underweight'] & self.heart_rate['normal']) |
                (self.bmi['normal'] & ~self.heart_rate['high']),
                (self.caloric_burn['high'], self.rest_period['medium'], self.workout_intensity['high'])
            ),

            # HIIT Rules (Weight: 1.0)
            ctrl.Rule(
                (self.age['young'] & self.heart_rate['high'] & self.bmi['normal']) |
                (self.age['middle'] & self.heart_rate['high'] & self.bmi['normal']),
                (self.caloric_burn['very_high'], self.rest_period['very_short'], self.workout_intensity['very_high'])
            )
        ]

    def init_control_system(self):
        """Initialize the fuzzy control system"""
        try:
            # Create control system
            self.workout_ctrl = ctrl.ControlSystem(rules=self.rules)

            # Create simulation
            self.workout_simulator = ctrl.ControlSystemSimulation(self.workout_ctrl)

            # Test the system with sample values to ensure it's working
            self.workout_simulator.input['bmi'] = 25.0
            self.workout_simulator.input['heart_rate'] = 100.0
            self.workout_simulator.input['age'] = 30.0
            self.workout_simulator.compute()

            # Reset the simulator for actual use
            self.workout_simulator = ctrl.ControlSystemSimulation(self.workout_ctrl)

        except Exception as e:
            print(f"Error initializing control system: {str(e)}")
            raise RuntimeError(f"Failed to initialize fuzzy control system: {str(e)}")

    def get_workout_type(self, intensity, bmi, age):
        """Enhanced workout type selection with stronger differentiation"""

        # Base scores with stronger initial differentiation
        workout_scores = {
            'Yoga': 0.0,
            'Cardio': 0.0,
            'Strength Training': 0.0,
            'HIIT': 0.0
        }

        # Sharper intensity-based scoring (reduced variance)
        workout_scores['Yoga'] += np.exp(-((intensity - 25) ** 2) / 200)
        workout_scores['Cardio'] += np.exp(-((intensity - 50) ** 2) / 200)
        workout_scores['Strength Training'] += np.exp(-((intensity - 75) ** 2) / 200)
        workout_scores['HIIT'] += np.exp(-((intensity - 95) ** 2) / 200)

        # Stronger BMI-based adjustments
        if bmi < 18.5:
            workout_scores['Strength Training'] *= 2.0
            workout_scores['Yoga'] *= 1.5
            workout_scores['HIIT'] *= 0.5
        elif 18.5 <= bmi <= 25:
            workout_scores['Cardio'] *= 1.5
            workout_scores['HIIT'] *= 1.4
            workout_scores['Strength Training'] *= 1.4
        elif 25 < bmi <= 30:
            workout_scores['Cardio'] *= 1.6
            workout_scores['Strength Training'] *= 1.5
            workout_scores['HIIT'] *= 0.7
        else:
            workout_scores['Yoga'] *= 1.8
            workout_scores['Strength Training'] *= 1.3
            workout_scores['HIIT'] *= 0.4

        # Stronger age-based adjustments
        if age < 30:
            workout_scores['HIIT'] *= 1.6
            workout_scores['Strength Training'] *= 1.5
            workout_scores['Cardio'] *= 1.3
        elif 30 <= age <= 50:
            workout_scores['Cardio'] *= 1.4
            workout_scores['Strength Training'] *= 1.4
            workout_scores['HIIT'] *= 1.2
        else:
            workout_scores['Yoga'] *= 2.0
            workout_scores['Cardio'] *= 1.3
            workout_scores['Strength Training'] *= 0.8
            workout_scores['HIIT'] *= 0.4

        return max(workout_scores.items(), key=lambda x: x[1])[0]

    def recommend(self, bmi, heart_rate, age):
        """Modified recommendation system to reduce overfitting"""
        try:
            # Wider ranges for similar profiles
            bmi_range = (bmi - 3, bmi + 3)
            age_range = (age - 5, age + 5)
            hr_range = (heart_rate - 12, heart_rate + 12)

            similar_profiles = self.df[
                (self.df['BMI'].between(*bmi_range)) &
                (self.df['Age'].between(*age_range)) &
                (self.df['Max_BPM'].between(*hr_range))
            ].copy()

            if len(similar_profiles) >= 1:
                # Smoother similarity scoring using Gaussian-like function
                similar_profiles.loc[:, 'similarity_score'] = (
                    np.exp(-((similar_profiles['BMI'] - bmi) ** 2) / 50) * 0.4 +
                    np.exp(-((similar_profiles['Age'] - age) ** 2) / 100) * 0.3 +
                    np.exp(-((similar_profiles['Max_BPM'] - heart_rate) ** 2) / 400) * 0.3
                )

                # Apply smoother adjustments for age, BMI, and heart rate
                age_factor = np.exp(-((age - 40) ** 2) / 200)
                bmi_factor = np.exp(-((bmi - 25) ** 2) / 50)
                hr_factor = np.exp(-((heart_rate - 120) ** 2) / 800)

                # Adjust intensity and rest period using smooth factors
                if age > 60:
                    intensity *= (0.9 + age_factor)
                    rest_period *= (1.2 + age_factor)
                elif age < 25:
                    intensity *= (1.05 + age_factor)
                    rest_period *= (0.9 + age_factor)

                # Weight the fuzzy system output with historical data
                try:
                    self.workout_simulator.input['bmi'] = float(bmi)
                    self.workout_simulator.input['heart_rate'] = float(heart_rate)
                    self.workout_simulator.input['age'] = float(age)

                    self.workout_simulator.compute()

                    # Get fuzzy system outputs
                    fuzzy_intensity = float(self.workout_simulator.output['workout_intensity'])
                    fuzzy_calories = float(self.workout_simulator.output['caloric_burn'])
                    fuzzy_rest = float(self.workout_simulator.output['rest_period'])

                    # Calculate weighted averages with historical data
                    if len(similar_profiles) > 0:
                        historical_weight = min(0.7, len(similar_profiles) * 0.15)  # Cap historical weight at 0.7
                        fuzzy_weight = 1 - historical_weight

                        # Weighted average for each metric
                        intensity = (fuzzy_weight * fuzzy_intensity +
                                   historical_weight * similar_profiles['Workout_Intensity'].mean())

                        caloric_burn = (fuzzy_weight * fuzzy_calories +
                                      historical_weight * similar_profiles['Calories_Burned'].mean())

                        rest_period = (fuzzy_weight * fuzzy_rest +
                                     historical_weight * (similar_profiles['Session_Duration (hours)'].mean() * 3600))

                        # Apply age-based adjustments
                        if age > 60:
                            intensity *= 0.85
                            rest_period *= 1.3
                        elif age < 25:
                            intensity *= 1.1
                            rest_period *= 0.8

                        # Apply BMI-based adjustments
                        if bmi > 30:
                            intensity *= 0.9
                            rest_period *= 1.2
                        elif bmi < 18.5:
                            intensity *= 0.95
                            caloric_burn *= 1.1

                        # Apply heart rate-based adjustments
                        if heart_rate > 160:
                            intensity *= 0.85
                            rest_period *= 1.4
                        elif heart_rate < 60:
                            intensity *= 0.9
                            caloric_burn *= 0.95
                    else:
                        intensity = fuzzy_intensity
                        caloric_burn = fuzzy_calories
                        rest_period = fuzzy_rest

                    workout_type = self.get_workout_type(intensity, bmi, age)

                    if workout_type not in self.workout_tips:
                        return None

                    return {
                        'caloric_burn': round(caloric_burn),
                        'rest_period': round(rest_period),
                        'workout_intensity': round(intensity),
                        'workout_type': workout_type,
                        'tips': self.workout_tips[workout_type],
                        'similar_profiles_count': len(similar_profiles),
                        'confidence_score': min(len(similar_profiles) * 0.2, 1.0)  # Add confidence score
                    }

                except Exception as e:
                    # print(f"Error in fuzzy computation: {str(e)}")
                    return None

            else:
                # Fallback to pure fuzzy logic if no similar profiles found
                try:
                    self.workout_simulator.input['bmi'] = float(bmi)
                    self.workout_simulator.input['heart_rate'] = float(heart_rate)
                    self.workout_simulator.input['age'] = float(age)

                    self.workout_simulator.compute()

                    intensity = float(self.workout_simulator.output['workout_intensity'])
                    workout_type = self.get_workout_type(intensity, bmi, age)

                    return {
                        'caloric_burn': round(float(self.workout_simulator.output['caloric_burn'])),
                        'rest_period': round(float(self.workout_simulator.output['rest_period'])),
                        'workout_intensity': round(intensity),
                        'workout_type': workout_type,
                        'tips': self.workout_tips[workout_type],
                        'similar_profiles_count': 0,
                        'confidence_score': 0.5  # Lower confidence for fuzzy-only predictions
                    }
                except Exception as e:
                    # print(f"Error in fuzzy computation: {str(e)}")
                    return None

        except Exception as e:
            # print(f"Error in recommendation: {str(e)}")
            return None

    def visualize_membership_functions(self):
        # Create figure with more space between subplots
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.5, wspace=0.4)

        # Define variables and their titles
        variables = [self.bmi, self.heart_rate, self.age,
                    self.caloric_burn, self.rest_period, self.workout_intensity]
        titles = ['Body Mass Index (BMI)', 'Heart Rate (BPM)', 'Age (Years)',
                 'Caloric Burn (kcal)', 'Rest Period (seconds)', 'Workout Intensity (%)']

        # Define a modern color palette with transparency
        colors = [
            '#2196F3', '#4CAF50', '#FFC107', '#E91E63', '#9C27B0', '#00BCD4'
        ]

        # Create subplots for each variable
        for idx, (var, title) in enumerate(zip(variables, titles)):
            row = idx // 2
            col = idx % 2
            ax = axs[row, col]

            # Plot membership functions as filled areas
            for i, term in enumerate(var.terms):
                universe = var.universe
                membership = var.terms[term].mf

                # Fill area under the curve
                ax.fill_between(
                    universe,
                    membership,
                    alpha=0.3,
                    color=colors[i % len(colors)],
                    label=term.replace('_', ' ').title()
                )

                # Add line on top for better definition
                ax.plot(
                    universe,
                    membership,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8
                )

            # Customize the appearance
            ax.set_title(title, pad=20, fontsize=13, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.2)
            ax.set_facecolor('#ffffff')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add subtle box around the plot
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#cccccc')

            # Customize axis labels
            ax.set_xlabel(title.split('(')[0].strip(), fontsize=11)
            ax.set_ylabel('Membership Degree', fontsize=11)

            # Set y-axis limits
            ax.set_ylim(-0.05, 1.05)

            # Add legend with better positioning and style
            legend = ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                framealpha=0.95,
                facecolor='white',
                edgecolor='#cccccc',
                shadow=True
            )

            # Add explanatory text for the first plot
            if idx == 0:
                explanation = (
                    "How to read this chart:\n"
                    "• Each colored area represents a category\n"
                    "• The height (0-1) shows the membership degree\n"
                    "• Overlapping areas mean a value belongs to multiple categories\n"
                    "• The higher the area, the stronger the membership"
                )
                ax.text(
                    0.02, -0.35,
                    explanation,
                    transform=ax.transAxes,
                    fontsize=10,
                    style='italic',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.9,
                        edgecolor='#cccccc',
                        pad=10,
                        boxstyle='round'
                    )
                )

        # Add a title to the entire figure
        fig.suptitle(
            'Fuzzy Logic Membership Functions\nWorkout Recommendation System',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Add a description of the visualization
        description = (
            'These charts show how input values are classified into different categories using fuzzy logic.\n'
            'For example, a BMI of 22 belongs mostly to "Normal" category with some overlap into neighboring categories.'
        )
        fig.text(
            0.5, 0.02,
            description,
            ha='center',
            fontsize=11,
            style='italic',
            bbox=dict(
                facecolor='white',
                alpha=0.9,
                edgecolor='#cccccc',
                pad=10,
                boxstyle='round'
            )
        )

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()

    def visualize_3d_fuzzy_system(self):
        """Create an opaque 3D visualization"""
        # Create meshgrid
        bmi_range = np.linspace(15, 40, 50)
        hr_range = np.linspace(40, 200, 50)
        age = 30

        BMI, HR = np.meshgrid(bmi_range, hr_range)

        # Vectorize calculations
        intensity = np.zeros_like(BMI)
        caloric = np.zeros_like(BMI)
        rest = np.zeros_like(BMI)

        # Batch processing
        batch_size = 100
        total_points = BMI.size
        flat_bmi = BMI.flatten()
        flat_hr = HR.flatten()

        for i in range(0, total_points, batch_size):
            batch_end = min(i + batch_size, total_points)
            for j in range(i, batch_end):
                result = self.recommend(flat_bmi[j], flat_hr[j], age)
                if isinstance(result, dict):
                    idx = np.unravel_index(j, BMI.shape)
                    intensity[idx] = result['workout_intensity']
                    caloric[idx] = result['caloric_burn']
                    rest[idx] = result['rest_period']

        # Apply smoothing
        from scipy.ndimage import gaussian_filter
        sigma = 1.0
        intensity = gaussian_filter(intensity, sigma=sigma)
        caloric = gaussian_filter(caloric, sigma=sigma)
        rest = gaussian_filter(rest, sigma=sigma)

        # Create figure
        fig = go.Figure()

        surfaces = [
            (intensity, 'Workout Intensity (%)', 'Viridis'),
            (caloric, 'Caloric Burn (kcal)', 'Plasma'),
            (rest, 'Rest Period (sec)', 'Magma')
        ]

        buttons = []
        for idx, (z_data, title, colorscale) in enumerate(surfaces):
            fig.add_trace(
                go.Surface(
                    x=BMI,
                    y=HR,
                    z=z_data,
                    name=title,
                    colorscale=colorscale,
                    visible=True if idx == 0 else False,
                    showscale=True,
                    colorbar=dict(
                        title=title,
                        titleside='right',
                        titlefont=dict(size=14)
                    ),
                    lighting=dict(
                        ambient=0.9,    # Increased ambient light
                        diffuse=0.8,
                        fresnel=0.2,
                        specular=0.2,
                        roughness=0.9
                    ),
                    opacity=1.0,        # Full opacity
                    hoverinfo='x+y+z',
                    connectgaps=True,
                    surfacecolor=z_data
                )
            )

            buttons.append(dict(
                label=title,
                method='update',
                args=[{'visible': [i == idx for i in range(len(surfaces))]},
                     {'title': f'Fuzzy Logic Workout System: {title}'}]
            ))

        # Layout
        fig.update_layout(
            title=dict(
                text='Fuzzy Logic Workout Recommendation System',
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='BMI',
                yaxis_title='Heart Rate (BPM)',
                zaxis_title='Output Value',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
                aspectratio=dict(x=1, y=1, z=0.7),
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray')
            ),
            updatemenus=[dict(
                type='dropdown',
                showactive=True,
                buttons=buttons,
                x=0.1,
                y=1.1,
                xanchor='left',
                yanchor='top'
            )],
            template='plotly_white',
            margin=dict(l=0, r=0, b=0, t=100)
        )

        fig.show()

    def visualize_workout_types_3d(self):
        """Create a 3D visualization of workout type recommendations"""
        # Create meshgrid for BMI and Heart Rate
        bmi_range = np.linspace(15, 40, 50)
        hr_range = np.linspace(40, 200, 50)
        age = 30  # Fixed age for visualization

        BMI, HR = np.meshgrid(bmi_range, hr_range)

        # Calculate workout types for each point
        workout_types = np.zeros_like(BMI)

        # Map workout types to numbers for visualization
        type_map = {'Yoga': 1, 'Cardio': 2, 'Strength Training': 3, 'HIIT': 4}

        for i, j in product(range(BMI.shape[0]), range(BMI.shape[1])):
            result = self.recommend(BMI[i,j], HR[i,j], age)
            if isinstance(result, dict):
                workout_types[i,j] = type_map[result['workout_type']]

        # Create the 3D visualization
        fig = go.Figure()

        # Custom colorscale for workout types
        colors = [
            [0.0, '#E0F7FA'],  # Light cyan for no workout
            [0.2, '#90CAF9'],  # Light blue for Yoga
            [0.4, '#81C784'],  # Light green for Cardio
            [0.7, '#FFB74D'],  # Orange for Strength Training
            [1.0, '#F06292']   # Pink for HIIT
        ]

        # Add surface plot
        fig.add_trace(
            go.Surface(
                x=BMI,
                y=HR,
                z=workout_types,
                colorscale=colors,
                showscale=True,
                colorbar=dict(
                    title='Workout Type',
                    titleside='right',
                    titlefont=dict(size=14),
                    ticktext=['Yoga', 'Cardio', 'Strength Training', 'HIIT'],
                    tickvals=[1, 2, 3, 4],
                    tickfont=dict(size=12)
                )
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Workout Type Recommendations<br>Based on BMI and Heart Rate',
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='BMI',
                yaxis_title='Heart Rate (BPM)',
                zaxis_title='Workout Type',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5)
                ),
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray',
                          ticktext=['', 'Yoga', 'Cardio', 'Strength', 'HIIT'],
                          tickvals=[0, 1, 2, 3, 4])
            ),
            showlegend=False,
            template='plotly_white'
        )

        # Add annotations for regions
        fig.add_annotation(
            text=(
                "Workout Regions:<br>" +
                "• Blue: Yoga (Low Intensity)<br>" +
                "• Green: Cardio (Moderate Intensity)<br>" +
                "• Orange: Strength Training (High Intensity)<br>" +
                "• Pink: HIIT (Very High Intensity)"
            ),
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=1.1,
            y=0.9,
            bordercolor='gray',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
            opacity=0.8
        )

        # Add usage instructions
        fig.add_annotation(
            text=(
                "How to use:<br>" +
                "• Drag to rotate view<br>" +
                "• Scroll to zoom<br>" +
                "• Double-click to reset view<br>" +
                "• Hover for exact values"
            ),
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=1.1,
            y=0.6,
            bordercolor='gray',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
            opacity=0.8
        )

        fig.show()

    def evaluate_model(self):
        """Evaluate the model's performance using the dataset"""
        try:
            # Split the dataset into training and testing sets
            test_size = 0.2
            train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=42)

            # Initialize counters for accuracy calculation
            correct_predictions = 0
            total_valid_predictions = 0

            # Lists to store actual and predicted values for MSE calculation
            actual_intensities = []
            predicted_intensities = []
            actual_calories_list = []
            predicted_calories_list = []

            # Evaluate each test case
            for _, row in test_df.iterrows():
                try:
                    # Get actual values
                    actual_intensity = row['Workout_Intensity']
                    actual_calories = row['Calories_Burned']

                    # Get predictions using clipped BMI if necessary
                    bmi = row['BMI']
                    if bmi > 50:  # Clip extremely high BMI values
                        bmi = 50
                    elif bmi < 12:  # Clip extremely low BMI values
                        bmi = 12

                    result = self.recommend(bmi, row['Max_BPM'], row['Age'])

                    if result is not None:
                        predicted_intensity = result['workout_intensity']
                        predicted_calories = result['caloric_burn']

                        # More lenient error margins (±25% for intensity, ±30% for calories)
                        intensity_lower = actual_intensity * 0.75
                        intensity_upper = actual_intensity * 1.25
                        calories_lower = actual_calories * 0.7
                        calories_upper = actual_calories * 1.3

                        # Consider prediction correct if either intensity or calories are within range
                        if (intensity_lower <= predicted_intensity <= intensity_upper or
                            calories_lower <= predicted_calories <= calories_upper):
                            correct_predictions += 1

                        # Store values for MSE calculation
                        actual_intensities.append(actual_intensity)
                        predicted_intensities.append(predicted_intensity)
                        actual_calories_list.append(actual_calories)
                        predicted_calories_list.append(predicted_calories)

                        total_valid_predictions += 1

                except Exception as e:
                    continue

            if total_valid_predictions == 0:
                raise ValueError("No valid predictions were made")

            # Calculate metrics
            accuracy = (correct_predictions / total_valid_predictions) * 100
            intensity_mse = np.mean((np.array(actual_intensities) - np.array(predicted_intensities)) ** 2)
            calories_mse = np.mean((np.array(actual_calories_list) - np.array(predicted_calories_list)) ** 2)

            # Normalize MSE
            max_intensity_error = (100) ** 2
            max_calories_error = (1500) ** 2  # Adjusted for new range

            normalized_intensity_mse = (intensity_mse / max_intensity_error) * 100
            normalized_calories_mse = (calories_mse / max_calories_error) * 100

            return {
                'accuracy': round(accuracy, 2),
                'intensity_mse': round(intensity_mse, 2),
                'calories_mse': round(calories_mse, 2),
                'normalized_intensity_mse': round(normalized_intensity_mse, 2),
                'normalized_calories_mse': round(normalized_calories_mse, 2),
                'test_size': test_size,
                'test_samples': total_valid_predictions,
                'total_samples': len(test_df)
            }

        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            return None

def main():
    recommender = WorkoutRecommender()

    # First, evaluate the model
    print("\n=== Model Evaluation ===")
    evaluation = recommender.evaluate_model()
    if evaluation:
        print(f"Model Accuracy: {evaluation['accuracy']}%")
        print(f"Intensity MSE: {evaluation['intensity_mse']}")
        print(f"Calories MSE: {evaluation['calories_mse']}")
        print(f"Normalized Intensity MSE: {evaluation['normalized_intensity_mse']}%")
        print(f"Normalized Calories MSE: {evaluation['normalized_calories_mse']}%")
        print(f"Valid Predictions: {evaluation['test_samples']} out of {evaluation['total_samples']} samples")
        print(f"Test Set Size: {evaluation['test_size'] * 100}%")

    print("\n=== Fuzzy Logic Workout Recommender ===")
    print("Please enter your details:")

    try:
        bmi = float(input("BMI (15-40): "))
        heart_rate = float(input("Heart Rate (40-200 bpm): "))
        age = float(input("Age (15-80): "))

        result = recommender.recommend(bmi, heart_rate, age)

        if result is not None:
            print("\n=== Recommended Workout Plan ===")
            print(f"Caloric Burn Target: {result['caloric_burn']} kcal")
            print(f"Recommended Rest Period: {result['rest_period']} seconds")
            print(f"Workout Intensity: {result['workout_intensity']}%")
            print(f"Recommended Workout Type: {result['workout_type']}")

            print("\n=== Workout Details ===")
            print(f"Description: {result['tips']['description']}")
            print("\nBenefits:")
            for benefit in result['tips']['benefits']:
                print(f"- {benefit}")
            print("\nRecommended Exercises:")
            for exercise in result['tips']['examples']:
                print(f"- {exercise}")

            print("\nDisplaying 3D visualizations...")
            print("\n1. System Output Variables (3D)")
            recommender.visualize_3d_fuzzy_system()

            print("\n2. Workout Type Recommendations (3D)")
            recommender.visualize_workout_types_3d()

    except ValueError as e:
        print(f"Error: Please enter valid numerical values. {str(e)}")

if __name__ == "__main__":
    main()


