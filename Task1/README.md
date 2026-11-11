# Cyclone Sensor Analytics

## 1. Project Overview

This project analyses historical industrial cyclone sensor data to identify operational states, detect abnormal process behaviour, investigate potential shutdown precursors, and forecast Cyclone Inlet Gas Temperature.

The project is divided into six phases:

1. Data Inspection and Preparation
2. Shutdown and Idle Detection
3. Feature Engineering and Operational State Clustering
4. Contextual Anomaly Detection and Root-Cause Investigation
5. Short-Horizon Forecasting
6. Cross-Phase Analysis and Operational Insights

The objective is to transform historical sensor readings into actionable operational insights that can support process monitoring, early-warning systems, and maintenance decisions.

## 2. Dataset Description

The dataset contains approximately 377,719 original records collected at five-minute intervals between January 2017 and August 2020.

It contains six cyclone sensor variables:

| Sensor | Description |
|---|---|
| `Cyclone_Inlet_Gas_Temp` | Cyclone inlet gas temperature |
| `Cyclone_Material_Temp` | Material temperature |
| `Cyclone_Outlet_Gas_draft` | Cyclone outlet gas draft |
| `Cyclone_cone_draft` | Cyclone cone draft |
| `Cyclone_Gas_Outlet_Temp` | Cyclone outlet gas temperature |
| `Cyclone_Inlet_Draft` | Cyclone inlet draft |

The original dataset also contains a timestamp column and non-numeric sensor fault/status values. Data preparation included numeric conversion, timestamp validation, missing-value investigation, and outlier analysis.

## 3. Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- OpenPyXL

Machine learning techniques:
- KMeans clustering
- Isolation Forest
- Random Forest Regression
- StandardScaler
- Persistence forecasting baseline

## 4. Project Structure

```text
Task1/
├── data/
│   └── raw/
│       └── data.xlsx
├── notebooks/
│   └── task1_analysis.ipynb
├── outputs/
│   ├── cleaned_data.csv
│   ├── data_quality_report.csv
│   ├── summary_statistics.csv
│   ├── shutdown_periods.csv
│   ├── cluster_assignments.csv
│   ├── clusters_summary.csv
│   ├── anomalous_periods.csv
│   ├── forecasts.csv
│   └── forecast_metrics.csv
└── plots/
```

Additional Phase 6 cross-analysis outputs are generated in the `outputs/` directory.

## 5. Environment Setup

### Prerequisites

- Python 3
- Git
- Visual Studio Code or Jupyter Notebook

### Create and activate the environment

From the project root:

```bash
python3 -m venv Task1/env1
source Task1/env1/bin/activate
```

### Install dependencies

```bash
pip install -r Task1/requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`Task1/notebooks/task1_analysis.ipynb`

Ensure the raw dataset is available at:

`Task1/data/raw/data.xlsx`

The raw dataset is excluded from Git and must be supplied locally.

## 6. Execution Workflow

Run the notebook sequentially from top to bottom.

| Phase | Analysis | Main Deliverables |
|---|---|---|
| Phase 1 | Data inspection, cleaning, missing-value handling, EDA | `cleaned_data.csv`, data quality report, summary statistics, EDA plots |
| Phase 2 | Shutdown and idle detection | `shutdown_periods.csv`, shutdown visualization |
| Phase 3 | Feature engineering and KMeans clustering | `cluster_assignments.csv`, `clusters_summary.csv`, clustering plots |
| Phase 4 | Cluster-specific anomaly detection and investigation | `anomalous_periods.csv`, five annotated anomaly plots |
| Phase 5 | One-hour-ahead temperature forecasting | `forecasts.csv`, `forecast_metrics.csv`, forecasting plots |
| Phase 6 | Cross-phase analysis, insights, and monitoring recommendations | Shutdown-state analysis, anomaly-rate analysis, consolidated findings |

## 7. Key Findings

### 7.1 Shutdowns and Operational States

- 135 shutdown events were identified.
- 99 shutdowns (approximately 73.3%) were immediately preceded by the Reduced Draft operational state.
- Reduced Draft represents approximately 2.3% of active operating time.

This indicates a strong historical association between Reduced Draft and subsequent shutdowns.

### 7.2 Anomaly Behaviour Across Operational States

Anomaly rates ranged from approximately 69 to 98 events per 1,000 active hours across the six operational states.

The results suggest that anomaly frequency is relatively uniform across states, while transitions into Reduced Draft provide a more distinctive shutdown-related signal.

### 7.3 Draft-Related Warning Signals

Among 114 anomalous events occurring within three hours before shutdowns, 40 implicated the inlet and outlet draft sensor pair.

This highlights the importance of monitoring draft instability as part of shutdown-risk assessment.

### 7.4 Independent Process-Quality Issues

Only approximately 3.4% of anomalies occurred within 60 minutes before a shutdown.

Most anomalies therefore represent other process-quality concerns, including thermal excursions, degraded heat recovery, and material-feed irregularities.

### 7.5 Forecasting Performance

The Random Forest model achieved an overall RMSE of 21.45 compared with 22.67 for the persistence baseline.

However, Random Forest performed worse during shutdown-overlapping forecast windows, highlighting the importance of accounting for changing operating conditions.

## 8. Recommended Monitoring Strategies

1. **Draft-weakening alert:** Monitor sustained Reduced Draft signatures while temperatures remain elevated.
2. **Joint draft anomaly alert:** Prioritize simultaneous inlet and outlet draft deviations.
3. **Possible blockage alert:** Monitor cone-draft weakening alongside outlet gas temperature drops.
4. **Material-feed alert:** Detect material temperature deviations independently of gas-side measurements.
5. **State-aware forecasting:** Reduce forecast confidence or blend models when operating conditions approach shutdown-related transitions.

These are proposed monitoring strategies based on historical analysis. They require further validation before production deployment.

## 9. Limitations

- Root-cause explanations are hypotheses based on sensor behaviour, not confirmed equipment failures.
- Historical associations do not establish causation.
- A global forecasting model may struggle during abrupt operating regime changes.
- Proposed alert thresholds require validation using additional operating data and domain-expert feedback.
- The analysis uses historical data and does not represent a deployed real-time monitoring system.

## 10. Conclusion

The six-phase analysis demonstrates how industrial sensor data can be used to identify operational states, detect contextual anomalies, investigate potential shutdown indicators, and forecast process temperatures.

The combined findings emphasize the importance of draft stability, operational-state transitions, and shutdown-aware forecasting in industrial process monitoring.