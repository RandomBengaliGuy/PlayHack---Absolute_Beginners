import json

def create_notebook(cells, filename):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in text.split('\n')]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in code.split('\n')]
    }

# EDA Notebook
eda_cells = []

eda_cells.append(md_cell("# Exploratory Data Analysis (EDA)\nIn this notebook, we will explore the PlayHack ML track dataset."))

eda_cells.append(code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set_theme(style="whitegrid")
'''))

eda_cells.append(md_cell("## 1. Load Data"))

eda_cells.append(code_cell('''# Load labels and metadata
labels = pd.read_csv('train_labels.csv')
metadata = pd.read_csv('athlete_metadata.csv')

# Load daily/hourly data
daily_activity = pd.read_csv('dailyActivity_merged.csv')
sleep_day = pd.read_csv('sleepDay_merged.csv')
weight_log = pd.read_csv('weightLogInfo_merged.csv')
training_sessions = pd.read_csv('training_sessions.csv')

hourly_steps = pd.read_csv('hourlySteps_merged.csv')
hourly_calories = pd.read_csv('hourlyCalories_merged.csv')
hourly_intensities = pd.read_csv('hourlyIntensities_merged.csv')
hourly_heartrate = pd.read_csv('hourlyHeartrate_merged.csv')

print(f"Labels shape: {labels.shape}")
print(f"Metadata shape: {metadata.shape}")
'''))

eda_cells.append(md_cell("## 2. Target Variable Analysis"))

eda_cells.append(code_cell('''# Distribution of injury (Task A target)
fig = px.pie(labels, names='injured_in_risk_window', title='Proportion of Injured Athletes in Risk Window')
fig.show()
'''))

eda_cells.append(code_cell('''# Distribution of onset_day_offset and recovery_duration (Task B targets)
injured_labels = labels[labels['injured_in_risk_window'] == 1]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(injured_labels['onset_day_offset'], bins=15, kde=True)
plt.title('Distribution of Onset Day Offset')
plt.xlabel('Day in Risk Window (31-60)')

plt.subplot(1, 2, 2)
sns.histplot(injured_labels['recovery_duration'], bins=20, kde=True, color='orange')
plt.title('Distribution of Recovery Duration')
plt.xlabel('Days of Recovery')
plt.tight_layout()
plt.show()
'''))

eda_cells.append(md_cell("## 3. Metadata Analysis"))

eda_cells.append(code_cell('''# Merge labels with metadata
df_meta = pd.merge(metadata, labels, on='athlete_id', how='left')
df_meta.head()
'''))

eda_cells.append(code_cell('''# Injury rate by sport
plt.figure(figsize=(10, 6))
sns.barplot(data=df_meta, x='sport', y='injured_in_risk_window', ci=None)
plt.title('Injury Rate by Sport')
plt.ylabel('Proportion Injured')
plt.xticks(rotation=45)
plt.show()
'''))

eda_cells.append(code_cell('''# Numeric features distribution vs Injury
numeric_cols = ['age', 'height_cm', 'weight_kg_baseline', 'years_playing', 'prior_season_injury_count']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df_meta, x='injured_in_risk_window', y=col)
    plt.title(f'{col} by Injury Status')
plt.tight_layout()
plt.show()
'''))

eda_cells.append(code_cell('''# Correlation Heatmap for metadata
plt.figure(figsize=(10, 8))
corr = df_meta[numeric_cols + ['injured_in_risk_window', 'onset_day_offset', 'recovery_duration']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap - Metadata & Targets')
plt.show()
'''))

eda_cells.append(md_cell("## 4. Daily Activity & Sleep Analysis"))

eda_cells.append(code_cell('''# Ensure common ID column name
daily_activity.rename(columns={'Id': 'athlete_id'}, inplace=True)
sleep_day.rename(columns={'Id': 'athlete_id'}, inplace=True)

# Aggregate daily activity per athlete
activity_agg = daily_activity.groupby('athlete_id').agg({
    'TotalSteps': 'mean',
    'TotalDistance': 'mean',
    'VeryActiveMinutes': 'mean',
    'SedentaryMinutes': 'mean',
    'Calories': 'mean'
}).reset_index()

df_activity = pd.merge(df_meta[['athlete_id', 'injured_in_risk_window']], activity_agg, on='athlete_id', how='inner')

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(data=df_activity, x='TotalSteps', hue='injured_in_risk_window', fill=True)
plt.title('Average Daily Steps Distribution')

plt.subplot(1, 2, 2)
sns.kdeplot(data=df_activity, x='VeryActiveMinutes', hue='injured_in_risk_window', fill=True)
plt.title('Average Very Active Minutes')
plt.tight_layout()
plt.show()
'''))

eda_cells.append(code_cell('''# Sleep efficiency
sleep_day['SleepEfficiency'] = sleep_day['TotalMinutesAsleep'] / sleep_day['TotalTimeInBed']
sleep_agg = sleep_day.groupby('athlete_id').agg({
    'TotalMinutesAsleep': 'mean',
    'SleepEfficiency': 'mean'
}).reset_index()

df_sleep = pd.merge(df_meta[['athlete_id', 'injured_in_risk_window']], sleep_agg, on='athlete_id', how='inner')

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_sleep, x='injured_in_risk_window', y='SleepEfficiency')
plt.title('Sleep Efficiency by Injury Status')
plt.show()
'''))

eda_cells.append(md_cell("## 5. Training Sessions Analysis"))

eda_cells.append(code_cell('''# Calculate duration of sessions
training_sessions['duration'] = training_sessions['end_hour'] - training_sessions['start_hour']
# Handle negative duration if crossing midnight (assuming 24h format)
training_sessions['duration'] = training_sessions['duration'].apply(lambda x: x if x >= 0 else x + 24)

train_agg = training_sessions.groupby('athlete_id').agg({
    'session_id': 'count',
    'duration': 'sum'
}).rename(columns={'session_id': 'total_sessions', 'duration': 'total_training_hours'}).reset_index()

df_train = pd.merge(df_meta[['athlete_id', 'injured_in_risk_window']], train_agg, on='athlete_id', how='left').fillna(0)

fig = px.scatter(df_train, x='total_sessions', y='total_training_hours', color='injured_in_risk_window', 
                 title='Total Sessions vs Total Training Hours', opacity=0.7)
fig.show()
'''))

eda_cells.append(md_cell("## 6. Time Series / Hourly Data (Example)"))

eda_cells.append(code_cell('''# Merge hourly HR and Steps
hourly_heartrate.rename(columns={'Id': 'athlete_id'}, inplace=True)
hourly_steps.rename(columns={'Id': 'athlete_id'}, inplace=True)

# Average heart rate over the day (0-23 hours) - simplified extraction from ActivityHour
try:
    hourly_heartrate['Hour'] = pd.to_datetime(hourly_heartrate['ActivityHour']).dt.hour
    hr_hourly = hourly_heartrate.groupby('Hour')['AvgHeartRate'].mean().reset_index()
    
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=hr_hourly, x='Hour', y='AvgHeartRate', marker='o')
    plt.title('Average Heart Rate Pattern Across the Day')
    plt.xticks(range(24))
    plt.show()
except Exception as e:
    print("Could not parse datetime, check format:", e)
'''))

create_notebook(eda_cells, "eda.ipynb")

# Model Notebook
model_cells = []

model_cells.append(md_cell("# Feature Engineering & Modeling\nIn this notebook, we'll build features from the 30-day observation window and train machine learning models for Task A and Task B."))

model_cells.append(code_cell('''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, mean_absolute_error, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
'''))

model_cells.append(md_cell("## 1. Data Loading & Preprocessing"))

model_cells.append(code_cell('''labels = pd.read_csv('train_labels.csv')
metadata = pd.read_csv('athlete_metadata.csv')
daily_activity = pd.read_csv('dailyActivity_merged.csv').rename(columns={'Id': 'athlete_id'})
sleep_day = pd.read_csv('sleepDay_merged.csv').rename(columns={'Id': 'athlete_id'})
training_sessions = pd.read_csv('training_sessions.csv')
'''))

model_cells.append(code_cell('''# Feature Engineering Function
def filter_observation_window(df, date_col, id_col='athlete_id'):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        start_dates = df.groupby(id_col)[date_col].min()
        df['day'] = (df[date_col] - df[id_col].map(start_dates)).dt.days + 1
        return df[df['day'] <= 30].copy()
    return df

def build_features(metadata, daily, sleep, training):
    df = metadata.copy()
    
    # Encode categoricals
    le = LabelEncoder()
    cat_cols = ['sport', 'gender', 'dominant_side', 'position']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    # Filter datasets to first 30 days (Observation Window)
    daily_obs = filter_observation_window(daily, 'ActivityDate')
    sleep_obs = filter_observation_window(sleep, 'SleepDay')
    training_obs = filter_observation_window(training, 'date')
    
    # Daily Activity Aggregations
    daily_agg = daily_obs.groupby('athlete_id').agg({
        'TotalSteps': ['mean', 'std', 'sum'],
        'TotalDistance': ['mean', 'max'],
        'VeryActiveMinutes': ['mean', 'sum'],
        'SedentaryMinutes': ['mean', 'std'],
        'Calories': ['mean', 'sum']
    })
    daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
    daily_agg = daily_agg.reset_index()
    
    # Sleep Aggregations
    sleep_obs['SleepEfficiency'] = sleep_obs['TotalMinutesAsleep'] / sleep_obs['TotalTimeInBed']
    sleep_agg = sleep_obs.groupby('athlete_id').agg({
        'TotalMinutesAsleep': ['mean', 'std'],
        'SleepEfficiency': ['mean', 'min']
    })
    sleep_agg.columns = ['sleep_' + '_'.join(col).strip() for col in sleep_agg.columns.values]
    sleep_agg = sleep_agg.reset_index()
    
    # Training Sessions Aggregations
    training_obs['duration'] = training_obs['end_hour'] - training_obs['start_hour']
    training_obs['duration'] = training_obs['duration'].apply(lambda x: x if x >= 0 else x + 24)
    train_agg = training_obs.groupby('athlete_id').agg({
        'session_id': 'count',
        'duration': ['sum', 'mean']
    })
    train_agg.columns = ['train_' + '_'.join(col).strip() for col in train_agg.columns.values]
    train_agg = train_agg.reset_index()
    
    # Merge all
    df = pd.merge(df, daily_agg, on='athlete_id', how='left')
    df = pd.merge(df, sleep_agg, on='athlete_id', how='left')
    df = pd.merge(df, train_agg, on='athlete_id', how='left')
    
    # Fill NAs
    df = df.fillna(0)
    
    # Drop team_id (or could encode it)
    if 'team_id' in df.columns:
        df = df.drop(columns=['team_id'])
        
    return df

X_all = build_features(metadata, daily_activity, sleep_day, training_sessions)
data = pd.merge(X_all, labels, on='athlete_id', how='inner')

print(f"Features shape: {data.shape}")
'''))

model_cells.append(md_cell("## 2. Task A: Classification (injured_in_risk_window)"))

model_cells.append(code_cell('''X = data.drop(columns=['athlete_id', 'injured_in_risk_window', 'onset_day_offset', 'recovery_duration'])
y_clf = data['injured_in_risk_window']

X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
'''))

model_cells.append(code_cell('''# Grid Search for XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

scorer = make_scorer(f1_score)
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring=scorer, n_jobs=-1, verbose=1)
grid_xgb.fit(X_train_sc, y_train)

print(f"Best XGB Params: {grid_xgb.best_params_}")
preds_xgb = grid_xgb.predict(X_test_sc)
print(f"XGB F1 Score: {f1_score(y_test, preds_xgb):.4f}")
'''))

model_cells.append(code_cell('''# CatBoost Classifier
cat_model = CatBoostClassifier(iterations=300, random_seed=42, verbose=0)
param_grid_cat = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_cat = GridSearchCV(cat_model, param_grid_cat, cv=3, scoring=scorer, n_jobs=-1)
grid_cat.fit(X_train, y_train) # Catboost handles unscaled better but scaled is fine

print(f"Best CatBoost Params: {grid_cat.best_params_}")
preds_cat = grid_cat.predict(X_test)
print(f"CatBoost F1 Score: {f1_score(y_test, preds_cat):.4f}")
'''))

model_cells.append(code_cell('''# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
param_grid_lgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=3, scoring=scorer, n_jobs=-1)
grid_lgb.fit(X_train_sc, y_train)

print(f"Best LGBM Params: {grid_lgb.best_params_}")
preds_lgb = grid_lgb.predict(X_test_sc)
print(f"LGBM F1 Score: {f1_score(y_test, preds_lgb):.4f}")
'''))

model_cells.append(code_cell('''# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, scoring=scorer, n_jobs=-1)
grid_rf.fit(X_train_sc, y_train)

print(f"Best RF Params: {grid_rf.best_params_}")
preds_rf = grid_rf.predict(X_test_sc)
print(f"RF F1 Score: {f1_score(y_test, preds_rf):.4f}")
'''))

model_cells.append(md_cell("## 3. Task B: Regression (Onset & Recovery)"))

model_cells.append(code_cell('''# Filter only injured athletes for regression training
injured_data = data[data['injured_in_risk_window'] == 1]
X_reg = injured_data.drop(columns=['athlete_id', 'injured_in_risk_window', 'onset_day_offset', 'recovery_duration'])
y_onset = injured_data['onset_day_offset']
y_recovery = injured_data['recovery_duration']

# Train test split for regression
X_train_r, X_test_r, yo_train, yo_test, yr_train, yr_test = train_test_split(
    X_reg, y_onset, y_recovery, test_size=0.2, random_state=42
)

# Model for Onset Day (XGBoost Regressor)
reg_onset = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
reg_onset.fit(X_train_r, yo_train)
preds_onset = reg_onset.predict(X_test_r)
mae_onset = mean_absolute_error(yo_test, preds_onset)
print(f"Onset Day MAE: {mae_onset:.2f}")

# Model for Recovery Duration (LightGBM Regressor)
reg_recovery = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
reg_recovery.fit(X_train_r, yr_train)
preds_recovery = reg_recovery.predict(X_test_r)
mae_recovery = mean_absolute_error(yr_test, preds_recovery)
print(f"Recovery Duration MAE: {mae_recovery:.2f}")
'''))

model_cells.append(md_cell("## 4. Final Pipeline Construction\nUse the best classification model and best regression models to make predictions for a test set or for submission."))

model_cells.append(code_cell('''# Assuming 'grid_lgb' gave best F1 (for example)
best_classifier = grid_lgb.best_estimator_

# If we had test_data without labels:
# X_test_unseen = build_features(test_metadata, test_daily, test_sleep, test_training)
# test_preds_clf = best_classifier.predict(X_test_unseen)
# test_preds_onset = reg_onset.predict(X_test_unseen)
# test_preds_recovery = reg_recovery.predict(X_test_unseen)

# sample_submission['injured_in_risk_window'] = test_preds_clf
# sample_submission['onset_day_offset'] = np.clip(np.round(test_preds_onset), 1, 30)
# sample_submission['recovery_duration'] = np.clip(np.round(test_preds_recovery), 1, None)
# sample_submission.to_csv('submission.csv', index=False)
'''))

create_notebook(model_cells, "model.ipynb")
print("Successfully generated eda.ipynb and model.ipynb")
