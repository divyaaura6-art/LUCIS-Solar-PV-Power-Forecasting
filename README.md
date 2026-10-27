# LUCIS: An Intelligent Solar Power Forecasting System Using Machine Learning and Deep Learning

### Short-Term Solar Photovoltaic Power Forecasting Using Historical Generation Data

## 1. Project Overview

LUCIS is an intelligent solar photovoltaic (PV) power forecasting system developed to predict the next 24 hours of solar power generation using historical power measurements.

Solar energy generation varies throughout the day due to changing environmental and atmospheric conditions. Accurate forecasting helps improve energy scheduling, power management, and grid planning.

The system uses seven days of historical hourly PV generation data to forecast the following 24 hours. It evaluates multiple machine learning and deep learning models to identify their forecasting capabilities and compare their performance.

The project focuses on data preprocessing, time-series forecasting, model comparison, evaluation, and next-day prediction visualization.

## 2. Project Objectives

- Develop a reliable short-term solar PV power forecasting system.
- Process and prepare historical solar generation data.
- Generate forecasting sequences using 168 hours of historical data.
- Predict the next 24 hours of solar power generation.
- Implement and compare machine learning and deep learning models.
- Evaluate model performance using MAE, RMSE, and R².
- Generate and visualize next-day solar power forecasts.

## 3. Dataset Description

The project uses historical solar generation data from a nominal 5 MW solar power plant located in Addakal, Mahabubnagar, Telangana, India.

**Dataset Details:**
- Historical coverage: March 2019 – August 2026
- Original sampling interval: 15 minutes
- Processed frequency: Hourly
- Total hourly slots: 65,736
- Valid hourly observations: 64,989
- Missing hourly observations: 747
- Installed capacity: 5 MW

### Data Preprocessing

The original 15-minute readings are converted into hourly observations. Incomplete hourly intervals are rejected instead of being incorrectly treated as zero-generation periods.

The forecasting dataset is structured using:
- Input sequence: Previous 168 hours (7 days)
- Prediction horizon: Next 24 hours

A chronological data split is used to prevent future information from leaking into model training:

- Training: Through 2024
- Validation: 2025
- Testing: January–August 2026

## 4. Models Implemented

Six forecasting approaches are evaluated:

**1. Random Forest**
A tree-based ensemble learning model used as the machine learning baseline.

**2. Original LSTM**
A Long Short-Term Memory neural network designed to learn temporal patterns in historical solar generation.

**3. Enhanced LSTM**
An alternative recurrent neural network architecture evaluated for forecasting performance.

**4. Attention LSTM**
An attention-based LSTM architecture designed to capture relevant temporal information.

**5. CNN-LSTM**
A hybrid architecture combining convolutional and recurrent neural networks.

**6. N-HiTS**
A neural hierarchical interpolation model designed for time-series forecasting.

## 5. Experimental Results

The models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---:|---:|---:|
| Random Forest | 0.1974 | 0.4107 | 0.9096 |
| Original LSTM | 0.2203 | 0.4309 | 0.9005 |
| N-HiTS | 0.2118 | 0.4427 | 0.8950 |
| CNN-LSTM | 0.2452 | 0.4482 | 0.8924 |
| Attention LSTM | 0.2762 | 0.4758 | 0.8787 |
| Enhanced LSTM | 0.2753 | 0.5089 | 0.8612 |

### Key Findings

- Random Forest achieved the best overall performance, with an RMSE of 0.4107 MW and an R² of 0.9096.
- Original LSTM performed best among the evaluated deep learning models in terms of RMSE and R².
- N-HiTS achieved the lowest MAE among the deep learning models.
- The results demonstrate the effectiveness of historical generation patterns for short-term solar forecasting.

## 6. Next-Day Forecasting

The trained Original LSTM model is used to demonstrate next-day forecasting for September 1, 2026.

The model uses the previous 168 hourly observations to generate 24 hourly power predictions.

**Forecast highlights:**
- Forecast horizon: 24 hours
- Maximum predicted generation: Approximately 2.892 MW
- Peak prediction time: 11:00 AM

The forecast demonstrates the expected daytime increase in solar generation, followed by a decline towards evening.

This is a historical forecasting demonstration and not a measured accuracy evaluation for September 1, 2026.

## 7. System Workflow

1. Historical solar generation data collection.
2. Conversion of 15-minute readings into hourly observations.
3. Missing-value validation and preprocessing.
4. Creation of 168-hour input sequences and 24-hour targets.
5. Chronological training, validation, and testing.
6. Training of machine learning and deep learning models.
7. Model evaluation and comparison.
8. Next-day solar power forecasting.
9. Visualization and storage of forecasting results.

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Joblib
- LSTM Neural Networks
- Random Forest
- CNN-LSTM
- N-HiTS
- Time-Series Forecasting

## 9. Limitations and Future Scope

The current forecasting models primarily use historical solar PV generation measurements rather than fully integrated weather forecasts.

Future improvements include:

- Integrating validated meteorological variables into the forecasting pipeline.
- Improving nighttime prediction consistency.
- Conducting more controlled comparisons across model training periods.
- Exploring additional forecasting architectures.
- Developing an interactive dashboard for solar power monitoring and prediction.
- Evaluating the system using prospective real-world forecasts.

## 10. Conclusion

LUCIS demonstrates an end-to-end approach to short-term solar photovoltaic power forecasting using machine learning and deep learning.

The comparative evaluation identifies Random Forest as the strongest overall model, while Original LSTM provides the best RMSE and R² performance among the tested deep learning approaches.

The project highlights the potential of data-driven forecasting techniques in supporting renewable energy planning and solar power management.

## 11. Project Team

- Hemavarshini Sivakumar
- Divyadarshini P

## 12. Disclaimer

This project is developed for academic and research purposes. Forecasts are based on historical solar generation data and should not be treated as guaranteed future power generation or operational grid predictions.
