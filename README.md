A data-driven web application built with **Streamlit** to monitor historical NHS Accident & Emergency (A&E) performance and proactively forecast future hospital capacity. 

This project was developed as a Final Year Computer Science Project to address the NHS capacity crisis. It abstracts complex data science into an accessible, interactive decision-support tool for clinical coordinators and hospital managers.

## Key Features

* **Interactive Historical Analytics:** Explore decade-long A&E attendances, admission pathways, and 4-hour target performances using dynamic Altair charts.
* **Two-Stage Predictive Pipeline:** * **Stage 1 (Attendances):** Predicts future patient volumes. The system features a **Dynamic Routing Engine** that automatically utilizes **SARIMA** for short-term tactical planning (1-3 months) and Meta's **Prophet** algorithm for long-term strategic forecasting (6+ months) to handle pandemic-era data volatility.
  * **Stage 2 (Admissions):** A multivariate **Linear Regression** cascade intercepts the forecasted attendances and converts them into actionable, physical hospital bed admission predictions.
* **Smart Grouping UI:** Automatically aggregates data into clean trendlines if a user selects too many Hospital Trusts, preventing visual clutter ("hairball" charts).
* **Accessible Design:** Custom CSS forces strict adherence to NHS Digital Service Manual accessibility standards (WCAG 2.1 AA), prioritizing high-contrast typography and colour-blind friendly visualizations.

## 🛠️ Tech Stack

* **Language:** Python
* **Frontend:** Streamlit, Altair, HTML/CSS
* **Data Engineering:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Statsmodels (SARIMA), Prophet

## Project Structure

* `streamlitApp.py`: The main Streamlit dashboard application and UI routing logic.
* `DataModule.ipynb`: The Jupyter Notebook containing the automated data engineering pipeline (extracting, cleaning, and standardizing raw NHS CSV files).
* `ModelTraining.ipynb`: The research notebook where hyperparameter tuning, ACF correlograms, and algorithmic testing were conducted.
* `finalData.csv`: The aggregated, clean dataset used to power the dashboard.

## 🚀 Installation & Setup

To run this dashboard locally on your machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
cd Your-Repo-Name
