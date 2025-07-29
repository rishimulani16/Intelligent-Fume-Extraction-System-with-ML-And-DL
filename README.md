# ⚙️ FES Pressure Prediction & Anomaly Detection System  
### 🏭 Real-Time ML/DL Pipeline for Fume Extraction(FES) System in Steel Manufacturing  
**Deployed And Currently Used At ArcelorMittal/Nippon Steel (AM/NS) - Steel Making Plant 2 (SMP2)**

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5873f0f-6ff7-4be3-8012-1021d6e66df2" alt="FES1" width="600"/>
</p>
<p align="center">High-level architecture of the Fume Extraction System (FES).</p>

---

## 🚀 Project Overview

This project aims to build a **real-time predictive system** for pressure monitoring in the **Fume Extraction System (FES)** using a blend of **machine learning and deep learning** techniques.

When the system detects that a predicted pressure value exceeds a predefined threshold, it **triggers alerts and alarms**, enabling **preventive maintenance** to avoid breakdowns and ensure plant safety.

🔧 **Currently deployed in real-time operations at AM/NS SMP2.**

---

## 🎯 Key Features

✅ **Real-Time Prediction**: API-based integration for live pressure forecasting  
🔗 **Chained Model Architecture**: Interconnected models for sequential pressure prediction  
🧠 **High-Accuracy Deep Learning Core**: 94.5% accuracy with optimized DNN  
🚨 **Automated Anomaly Detection**: Built-in threshold alert system  
🏭 **Industrial-Scale Application**: Integrated into large-scale steel plant operations  

---

## 🛠️ Technical Stack

| Component     | Technology                          |
|---------------|-------------------------------------|
| Language      | Python                              |
| Libraries     | Scikit-learn, TensorFlow, Keras     |
| Deployment    | Flask API                           |
| Tools         | Jupyter, Anaconda, Conda            |

---

## 📊 Input Features

21 real-time sensor readings were used, including:

- Process Mode  
- Oxygen Consumption – Top  
- Furnace Power  
- Combustion Chamber Pressure  
- Suction & Canopy Damper Positions  
- Temperatures: FDC Inlet/Outlet, BagHouse Inlet (Upper & Lower)  
- Differential Pressures: Across FDC & BagHouse  
- ID Fan Speeds & Currents  
- Gas Mixture Inlet Pressure, FDC Outlet Pressure  

---

## 🎯 Target Variables (Predicted in Sequence)

1. Combustion Chamber Pressure  
2. FDC Outlet Pressure  
3. Pressure at Gas Mixture Inlet  
4. Pressure at BagHouse Inlet  
5. Pressure at ID Fan Inlet

Each prediction feeds into the next stage using a **chained model structure**.

---

## 🧠 Modeling Pipeline

| Stage                | Model Type             | Accuracy |
|----------------------|------------------------|----------|
| Baseline             | Linear Regression      | 41%      |
| Non-Linear Algorithm | Linear Regression      | 77%      |
| Non-Linear Boost     | Random Forest          | 89%      |
| Advanced Ensemble    | Voting/Stacking        | 93%      |
| Deep Learning (DNN)  | Deep Neural Network    | **94.5%** |

🔬 Final DNN Features:
- Multi-layer architecture  
- Batch Normalization  
- Dropout & L2 Regularization  
- Optimized with Adam optimizer  

---

## 🧩 System Architecture & Deployment

### 🔄 Real-Time Pipeline (via Flask)

1. **Data Ingestion**: API receives real-time sensor input  
2. **Preprocessing**: Data scaling & transformation  
3. **Chained Prediction**: Pressure predicted in sequence by 5 models  
4. **Anomaly Detection**: Final pressure compared to threshold  
5. **Alerting System**: Triggers warning/alarm if pressure exceeds safe limits  

🔧 Deployment guided by **Amerendra Sir**, integrated into the operational infrastructure of SMP2.

---

## 📈 Results & Impact

✅ Significantly reduced **downtime risk**  
✅ Improved **plant stability and safety**  
✅ Seamless integration into **AM/NS Steel operations**  
✅ Valuable learnings in **ML/DL**, **system deployment**, and **domain expertise**

---

## 🔮 Future Work

The current system provides highly accurate, real-time pressure predictions and anomaly detection. Future enhancements will expand the system’s capabilities to include time-series analysis and failure forecasting.

### 📊 Trend Prediction
- Analyze historical patterns in pressure, temperature, and fan speeds  
- Use statistical methods and moving averages to understand long-term deviations  
- Visualize system behavior over time to aid operational planning

### 🛠️ Predictive Maintenance
- Develop a model to **predict time-to-failure** for key components  
- Estimate **remaining useful life (RUL)** based on current trends and historical failures  
- Identify and alert on **expected anomaly windows**

### ⏳ Time-Series Modeling
Implement **deep learning-based forecasting models**:
- **LSTM (Long Short-Term Memory Networks)**  
   - Capable of learning temporal dependencies  
   - Suitable for pressure trends, failure signals, and sequential sensor readings  

> These upgrades will help transition from **reactive to predictive maintenance**, reducing downtime and enhancing plant efficiency.

---

## 🧪 Installation & Usage

### 🔍 Prerequisites
- Python 3.8+
- Anaconda or Miniconda

### ⚙️ Setup

```bash
git clone https://github.com/rishimulani16/Intelligent-Fume-Extraction-System-with-ML-And-DL.git
cd Intelligent-Fume-Extraction-System-with-ML-And-DL
conda env create -f environment.yml
conda activate fes_env
