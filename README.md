# Industrial Sensor Anomaly Detection

*Detect anomalies in industrial sensor data using machine learning.*



## Overview

This project analyses industrial sensor readings to identify anomalies that may indicate potential faults or abnormal machine behavior. It uses the **AI4I-style sensor dataset** containing time-stamped readings for multiple sensors.



The workflow demonstrates a full data science pipeline:

- Data cleaning and preprocessing  

- Exploratory Data Analysis (EDA)  

- Anomaly detection using Isolation Forest  

- Visualisation of anomalies  

- Insights for predictive maintenance  



Anomaly detection is critical for industrial monitoring, reducing unplanned downtime, and improving operational efficiency.



---



## Tools & Technologies

- **Python**  

- **Pandas, NumPy** for data manipulation  

\- \*\*Matplotlib, Seaborn\*\* for visualisation  

\- \*\*Scikit-learn\*\* for machine learning  

\- \*\*Jupyter Notebook\*\* for analysis  







---



## Key Steps Performed



**Data Cleaning & Preprocessing**

&nbsp;  - Split combined column into `Timestamp`, `SensorId`, and `Value`  

&nbsp;  - Pivoted the dataset for wide-format analysis  

&nbsp;  - Scaled sensor readings  



 **Exploratory Data Analysis (EDA)**

&nbsp;  - Histograms and distributions of sensor readings  



 **Anomaly Detection**

&nbsp;  - Trained Isolation Forest on scaled sensor data  

&nbsp;  - Labelled anomalies and normal points  

&nbsp;  - Visualised anomalies across sensor readings  



4.**Results & Insights**

&nbsp;  - Count and proportion of anomalies detected  

&nbsp;  - Plots highlighting unusual sensor behaviour  



---



## Results Summary

- Anomalies in sensor readings were successfully identified using Isolation Forest.  

- Visualisation provides insights into unusual patterns that may indicate machine faults.  

- This workflow can support predictive maintenance and operational monitoring.



---



## Next Steps

- Test alternative anomaly detection methods (One-Class SVM, Autoencoders)  

- Integrate real-time monitoring and alerting  

- Analyse anomalies to confirm true equipment faults  

- Deploy a monitoring dashboard using Streamlit or similar tools  



---





