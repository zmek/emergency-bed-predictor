import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from patientflow.load import load_data
from patientflow.prepare import create_temporal_splits, get_snapshots_at_prediction_time
from patientflow.train.emergency_demand import train_all_models
from patientflow.prepare import prepare_snapshots_dict
from patientflow.aggregate import get_prob_dist
from patientflow.viz.prob_dist_plot import prob_dist_plot
from patientflow.viz.qq_plot import qq_plot

# set up session state
if "plots" not in st.session_state:
    st.session_state.plots = {}


def generate_and_store_plot(plot_function, plot_key, *args, **kwargs):
    """Helper function to generate and store plots in session state"""
    try:
        fig = plot_function(*args, **kwargs)
        st.session_state.plots[plot_key] = fig
        return fig
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None


def main():
    st.title("Predict emergency bed demand in the next 8 hours")
    st.header("Using a simple Machine Learning model trained on your data.")

    # Add creator information
    st.markdown(
        """
    <style>
    .creator-info {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        background-color: #f0f2f6;
        font-size: 0.8em;
        border-top-left-radius: 5px;
    }
    </style>
    <div class='creator-info'>
    Created by Dr Zella King | Clinical Operational Research Unit, UCL | zella.king@ucl.ac.uk
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Introduction text
    st.markdown(
        """
    This tool predicts the number of emergency patients who will need a hospital bed in the next 8 hours. 
    You give it data about the patients in an Emergency Department (ED) at 09.30, and it will predict how many emergency beds you'll need by 17.30. 
    The predictions cover beds needed for patients in the ED at 09.30, and people who haven't arrived by that time. It applies ED 4-hour targets to all patients' arrival times.
    As part of this demo, you will train the model on a dataset of past visits to an ED. A fake dataset is provided.  
    

    Note: this tool is for demonstration purposes only. To use a tool like this for real, you would need to run this code in your own secure environment, prepare your data carefully, train the models, 
    and validate that the trained models perform to your satisfaction. 
    """
    )

    st.subheader("Step 1: Loading the data")

    # File upload
    st.markdown(
        """For this demonstration, we'll use a sample dataset of ED visits that contains 9 months of made-up data.
        It includes both patients who were admitted to a ward and those who were discharged from the ED.
        There is one row for each ED visit and snapshot date, with columns summarising what was known about the patient at that time.<br><br>
        Click the button below to load the data."""
    )

    if st.button("Load sample ED visits data"):
        try:
            st.info("Loading ED visits data from data folder")
            app_dir = Path(__file__).parent

            ed_visits = load_data(
                "data",
                file_name="ed_visits_minimal.csv",
                index_column="snapshot_id",
                eval_columns=[
                    "prediction_time",
                    "consultation_sequence",
                    "final_sequence",
                ],
                home_path=app_dir,
            )
            ed_visits["snapshot_date"] = pd.to_datetime(
                ed_visits["snapshot_date"]
            ).dt.date

            inpatient_arrivals = pd.read_csv(
                "data/inpatient_arrivals.csv", parse_dates=["arrival_datetime"]
            )
            # inpatient_arrivals = load_data(
            #     "data",
            #     file_name="inpatient_arrivals.csv",
            #     index_column="arrival_datetime",
            #     home_path=app_dir,
            # )

            # More robust date handling
            start_date = ed_visits.snapshot_date.min()
            end_date = ed_visits.snapshot_date.max()
            num_days = len(ed_visits.snapshot_date.unique())

            st.write(
                f"""The dataset starts on {start_date.strftime("%-d %B %Y")} and ends on {end_date.strftime("%-d %B %Y")}, 
                        and contains {len(ed_visits):,} ED visits over {num_days} days."""
            )

        except Exception as e:
            st.error(f"Error loading or processing the data: {str(e)}")
            return

        st.subheader("Step 2: Training the model")

        # Calculate the split points in days
        training_days = int(num_days * 0.7)
        validation_days = int(num_days * 0.1)
        # test_days will be the remainder

        # Calculate the start dates for each set
        start_training_set = start_date
        start_validation_set = start_date + pd.Timedelta(days=training_days)
        start_test_set = start_validation_set + pd.Timedelta(days=validation_days)
        end_test_set = end_date

        st.write(
            """Training the model, using the first 70% of the days in the uploaded file for training, and the next 10% for tuning the model. 
                    The remaining 20% will be saved to test the performance on the model on unseen data."""
        )

        # set up ordinal mappings
        ordinal_mappings = {
            "latest_obs_manchester_triage_acuity": [
                "Blue",
                "Green",
                "Yellow",
                "Orange",
                "Red",
            ],
        }
        # specify columns to exclude from training data
        exclude_from_training_data = [
            "snapshot_date",
            "prediction_time",
            "visit_number",
            "specialty",
            "consultation_sequence",
            "final_sequence",
        ]

        # Train the model

        model_metadata, models = train_all_models(
            ed_visits,
            start_training_set,
            start_validation_set,
            start_test_set,
            end_test_set,
            inpatient_arrivals,
            prediction_times=[(9, 30)],
            prediction_window=8,
            yta_time_interval=15,
            epsilon=0.00001,
            grid_params={"n_estimators": [30]},
            exclude_columns=exclude_from_training_data,
            ordinal_mappings=ordinal_mappings,
            uclh=False,
            random_seed=42,
            save_models=False,
            test_realtime=False,
        )

        st.write(f"Model trained successfully.")

        st.subheader("Step 3: Evaluating the model on a test set")
        st.write(
            """For each date in the test set, we'll load in the patients in the ED at 09.30, 
                 and compare the model's predictions with the actual number of patients who were admitted to a ward."""
        )
        # create the test set
        _, _, test_visits = create_temporal_splits(
            ed_visits.reset_index().copy(),
            start_training_set,
            start_validation_set,
            start_test_set,
            end_test_set,
            col_name="snapshot_date",
        )

        # get X_test and y_test at the chosen prediction time for input into the admissions model
        X_test, y_test = get_snapshots_at_prediction_time(
            test_visits,
            prediction_time=(9, 30),
            exclude_columns=exclude_from_training_data,
            single_snapshot_per_visit=False,
        )

        snapshots_dict = prepare_snapshots_dict(
            test_visits[(test_visits.prediction_time == (9, 30))]
        )
        first_record_key = list(snapshots_dict.keys())[0]

        # get probability distribution for this time of day
        with st.spinner("Calculating probability distributions for test set..."):
            prob_dist = get_prob_dist(
                snapshots_dict,
                X_test,
                y_test,
                model=models["admissions"]["admissions_0930"],
            )
            st.success("Probability distributions calculated successfully!")

        st.write(
            "Below is the probability distribution for the number of beds needed by patients in ED at 09:30 on the first record in the test set."
        )

        title_ = f"Probability distribution for number of beds needed by patients in ED at 09:30 on {first_record_key}"
        prob_dist_first_record = generate_and_store_plot(
            prob_dist_plot,
            "prob_dist_first_record",
            prob_dist_data=prob_dist[first_record_key]["agg_predicted"],
            title=title_,
            include_titles=True,
            return_figure=True,
        )

        # Add vertical line for observed value
        if prob_dist_first_record:
            observed_value = prob_dist[first_record_key]["agg_observed"]
            prob_dist_first_record.axes[0].axvline(
                x=observed_value,
                color="red",
                linestyle="--",
                label=f"Actual number of beds needed: {observed_value}",
            )
            prob_dist_first_record.axes[0].legend()
            st.pyplot(prob_dist_first_record)

        # We now show a plot of the performance of the model on the whole test set
        st.write(
            """Below is a plot that helps us to evaluate the performance of the model on the whole test set period.
            If the model performed well, the line will be close to the diagonal line"""
        )
        title = f"Probability distribution for number of beds needed by patients in ED at 09:30 on {first_record_key}"
        prob_dist_first_record = generate_and_store_plot(
            prob_dist_plot,
            "prob_dist_first_record",
            prob_dist_data=prob_dist[first_record_key]["agg_predicted"],
            title=title,
            include_titles=True,
            return_figure=True,
        )
        title = "Q-Q Plot for emergency demand predictions at 09:30"
        qq_plot_test_set = generate_and_store_plot(
            qq_plot,
            "qq_plot_test_set",
            snapshots_dict.keys(),
            prob_dist,
            title,
            return_figure=True,
            figsize=(3, 3),
        )
        if qq_plot_test_set:
            st.pyplot(qq_plot_test_set)


if __name__ == "__main__":
    main()
