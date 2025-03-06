# undelayed-demand: A streamlit app for visualising patterns of emergency demand by hour of day. 

Welcome to the undelayed-demand repository, which is designed to support hospital bed management through predictive modelling. I'm Zella King, a health data scientist in the Clinical Operational Research Unit (CORU) at University College London. Since 2020, I have worked with University College London Hospital (UCLH) NHS Trust on practical tools to improve patient flow through the hospital.

## What is this repository for? 

One reason why Emergency Departments (EDs) fail to meet admissions targets is because beds are not available at the time patients need to be admitted. UCLH wanted to understand their "undelayed" demand over the course of a day - that is, when beds would be needed if patients were processed within the 4-hour target time for ED. 

I created an application for an improvement project for UCLH, which was focused on their emergency patient pathway. The app show operations managers when they need to have bed ready for incoming patients.

The app itself is available on Streamlit Community [here](https://undelayed-demand-2jlyfpul5sawfbxd9tgz5n.streamlit.app/). You can upload a file of arrival datetimes of patients later admitted to your ED, and specify your ED targets. The app will generate a series of charts showing your patterns of demand. 

Follow the instructions below to set up a locally hosted version of the same Streamlit app. 




# Local Setup Instructions

## Prerequisites
- Python 3.10 or higher
- Git

## Setup Options

### Option 1: Using Conda (Recommended)
1. **Install Conda** 
   - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)

2. **Clone the repository**
   ```bash
   git clone https://github.com/zmek/undelayed-demand.git
   cd undelayed-demand
   ```

3. **Create and activate a virtual environment**
   ```bash
   conda create -n streamlit-app python=3.10
   conda activate streamlit-app
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Python venv
1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

After completing either setup option above:
```bash
streamlit run app.py
```

The app should automatically open in your default web browser at http://localhost:8501. If it doesn't, you can manually open this URL.

## Troubleshooting

- If you see import errors, ensure all dependencies are installed correctly
- For Conda users: If you see "(base)" and "(streamlit-app)" both showing in your terminal:
  ```bash
  conda deactivate  # Run this first to deactivate base
  conda activate streamlit-app  # Then reactivate your environment
  ```
- For Python venv users: Make sure you see "(venv)" at your command prompt, indicating the virtual environment is active
