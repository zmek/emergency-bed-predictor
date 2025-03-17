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
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor

# set up session state
if "plots" not in st.session_state:
    st.session_state.plots = {}
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


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
    st.title("Predict emergency bed demand")
    st.header(
        "The time is 09:30, and you want to know how many emergency beds you'll need by 17.30."
    )

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
    University College London Hospital (UCLH) has a tool that predicts their 8-hour bed demand, by specialty, assuming ED 4-hour targets are met. 
    Their Site Operations team refer to it daily. 
    
    This website is a demo of the tool, with fake data. 

    Once it has been set up with data from your hospital, it could be fed with data about patients in your ED today, and it will predict the number of beds you'll need by 17.30. 
    Based on targets for 4-hour performance that you set, it could show you the number of beds needed if you want to meet those targets.  
    
    Below, I show the tool set-up process. I then give it some fake 'real-time' data (the patients in my ED at 09:30 today) and ask it to make predictions.
    """
    )

    st.subheader("Step 1: Loading data on past ED visits")

    # File upload
    st.markdown(
        """The first step is to load data from the past about all patients who were in the Emergency Department (ED) at 09.30. 
        Here I'm using 9 months of made-up data. 
        The dataset includes all patients who were in the ED at 09:30 over the period, including patients later admitted to a ward and those who were later discharged.
        There is one row for each ED visit on each date, with columns summarising what was known about the patient at that time.
        
        
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
                "data/inpatient_arrivals.csv",
                parse_dates=["arrival_datetime"],
                date_parser=lambda x: pd.to_datetime(x, utc=True),
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
            st.write(
                """ 
                        
                        Click the button below to view the data. Scrolling to the right, you'll find a column called is_admitted, 
                        which tells you whether the patient was admitted to a ward later."""
            )

            st.session_state.data_loaded = True
            st.session_state.ed_visits = ed_visits
            st.session_state.inpatient_arrivals = inpatient_arrivals
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.session_state.num_days = num_days

        except Exception as e:
            st.error(f"Error loading or processing the data: {str(e)}")
            return

    # Only show the training section if data is loaded
    if st.session_state.data_loaded:
        st.subheader("Step 2: Training the model")

        st.write(
            """From the data, the tool learns how a patient's characteristics - such as their age, lab results, vital signs or referrals made while in the ED - relate to their likelihood of admission under a specialty. 
    This process is called 'training'. I use 80% of the data for training, and save the rest to test the performance of the model later."""
        )

        # Calculate the split points in days
        training_days = int(st.session_state.num_days * 0.7)
        validation_days = int(st.session_state.num_days * 0.1)
        # test_days will be the remainder

        # Calculate the start dates for each set
        start_training_set = st.session_state.start_date
        start_validation_set = st.session_state.start_date + pd.Timedelta(
            days=training_days
        )
        start_test_set = start_validation_set + pd.Timedelta(days=validation_days)
        end_test_set = st.session_state.end_date

        st.session_state.start_training_set = start_training_set
        st.session_state.start_validation_set = start_validation_set
        st.session_state.start_test_set = start_test_set
        st.session_state.end_test_set = end_test_set

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
        if st.button("Train the model"):
            with st.spinner("Training the model"):
                model_metadata, models = train_all_models(
                    st.session_state.ed_visits,
                    st.session_state.start_training_set,
                    st.session_state.start_validation_set,
                    st.session_state.start_test_set,
                    st.session_state.end_test_set,
                    st.session_state.inpatient_arrivals,
                    prediction_times=[(9, 30)],
                    prediction_window=8 * 60,
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

                # Save only the new variables in session state
                st.session_state.model_metadata = model_metadata
                st.session_state.models = models
                st.session_state.exclude_from_training_data = exclude_from_training_data
                st.session_state.model_trained = True
                st.success(f"Model trained successfully.")

    # Only show the predictions section if model is trained
    if st.session_state.model_trained:
        st.subheader("Step 3: Getting some real-time predictions")

        st.write(
            "Now let's give the model some real-time data about patients in the ED at 09:30 today, and ask it to make predictions."
        )

        if st.button("Show me some real-time predictions"):

            # create the test set
            _, _, test_visits = create_temporal_splits(
                st.session_state.ed_visits.reset_index().copy(),
                st.session_state.start_training_set,
                st.session_state.start_validation_set,
                st.session_state.start_test_set,
                st.session_state.end_test_set,
                col_name="snapshot_date",
            )

            # Get X_test and y_test using session state variables
            X_test, y_test = get_snapshots_at_prediction_time(
                test_visits,
                prediction_time=(9, 30),
                exclude_columns=st.session_state.exclude_from_training_data,
                single_snapshot_per_visit=False,
            )

            snapshots_dict = prepare_snapshots_dict(
                test_visits[(test_visits.prediction_time == (9, 30))]
            )

            last_record_key = list(snapshots_dict.keys())[-1]

            last_prob_dist = get_prob_dist(
                {last_record_key: snapshots_dict[last_record_key]},
                X_test,
                y_test,
                model=st.session_state.models["admissions"][
                    "admissions_0930"
                ].calibrated_pipeline,
            )

            title_ = f"Probability distribution for number of beds needed by patients in ED at 09:30 on {last_record_key}"
            prob_dist_last_record = generate_and_store_plot(
                prob_dist_plot,
                "prob_dist_first_record",
                prob_dist_data=last_prob_dist[last_record_key]["agg_predicted"],
                title=title_,
                include_titles=True,
                return_figure=True,
            )

            # Add vertical line for observed value
            if prob_dist_last_record:
                # observed_value = last_prob_dist[last_record_key]["agg_observed"]
                # prob_dist_last_record.axes[0].axvline(
                #     x=observed_value,
                #     color="red",
                #     linestyle="--",
                #     label=f"Actual number of beds needed: {observed_value}",
                # )
                # prob_dist_last_record.axes[0].legend()
                st.pyplot(prob_dist_last_record)

                st.session_state.prob_dist_generated = True
                st.session_state.last_prob_dist = last_prob_dist
                st.session_state.last_record_key = last_record_key

        # Move these blocks OUTSIDE the "Show me predictions" button (dedent)
        if (
            "prob_dist_generated" in st.session_state
            and st.session_state.prob_dist_generated
        ):
            st.subheader("Step 4: Confirm your ED targets")
            st.write(
                """The numbers above only patients currently in the ED. 
                Over the 8 hours between 09:30 and 17:30 there will be demand from patients who have not yet arrived.
                If your ED is hitting its targets, some of these patients will also need to be admitted by 17:30."""
            )

            st.write(
                """You have the option to specify your own ED targets. The default is 80% of patients being admitted within 4 hours and 99% within 12 hours. Use the sidebar to change this."""
            )

            # Sidebar controls for ED performance
            st.sidebar.header("Your aspirations for ED performance")
            st.sidebar.subheader("Main target")
            x1 = st.sidebar.number_input("Main target: Hours since ED arrival", value=4)
            y1 = (
                st.sidebar.number_input(
                    "Main target: Percentage of patients processed", value=80
                )
                / 100
            )

            st.sidebar.subheader("Mop-up target")
            x2 = st.sidebar.number_input(
                "Mop-up target: Hours since ED arrival", value=12
            )
            y2 = (
                st.sidebar.number_input(
                    "Mop-up target: Percentage of patients processed", value=99
                )
                / 100
            )

            st.markdown(
                f"Please confirm your ED performance targets:<br>"
                f"- **Main target:** Process {y1*100:.0f}% of admitted patients within {x1} hours<br>"
                f"- **Mop-up target:** Process {y2*100:.0f}% of admitted patients within {x2} hours",
                unsafe_allow_html=True,
            )

            if st.button("Confirm your targets"):
                st.session_state.ed_targets = {
                    "main_target_hours": x1,
                    "main_target_percent": y1,
                    "mopup_target_hours": x2,
                    "mopup_target_percent": y2,
                }
                st.session_state.targets_confirmed = True
                st.success("Targets confirmed and saved")

                # Move the calculations here so they update when targets are confirmed
                prediction_context = {"medical": {"prediction_time": (9, 30)}}
                yta_model = st.session_state.models["yet_to_arrive_8_hours"]
                targets = st.session_state.ed_targets
                yta_preds = yta_model.predict(
                    prediction_context,
                    targets["main_target_hours"],
                    targets["main_target_percent"],
                    targets["mopup_target_hours"],
                    targets["mopup_target_percent"],
                )

                title_ = f"Probability distribution for number of medical beds needed by 17:30 for patients arriving after 09:30 on {st.session_state.last_record_key}"
                prob_dist_yta_medical = generate_and_store_plot(
                    prob_dist_plot,
                    "prob_dist_yta_medical",
                    prob_dist_data=yta_preds["medical"],
                    title=title_,
                    include_titles=True,
                    return_figure=True,
                )

                if prob_dist_yta_medical:
                    st.pyplot(prob_dist_yta_medical)


if __name__ == "__main__":
    main()
