# Task 1 — Cyclone Sensor Analytics

## Objective
Analyze cyclone machine sensor data to identify operational states, detect contextual anomalies, forecast inlet gas temperature, and derive operational insights.

## Dataset
- Source: Provided Excel dataset
- File: `data/raw/data.xlsx`
- Sheet: `internship-data-1`
- Sampling interval: 5 minutes
- Sensor variables: 6

## Analysis workflow
1. Data preparation and validation
2. Exploratory data analysis
3. Shutdown and idle detection
4. Feature engineering
5. Operational-state clustering using KMeans
6. Contextual anomaly detection
7. Root-cause investigation
8. One-hour-ahead inlet gas temperature forecasting
9. Operational insights and recommendations

## Environment setup

Activate the environment:

    source Task1/env1/bin/activate

Install dependencies:

    pip install -r Task1/requirements.txt

Deactivate:

    deactivate

## Directory structure
- `data/raw/` — Original dataset
- `plots/` — Analysis visualizations
- `outputs/` — Processed data and analysis results
