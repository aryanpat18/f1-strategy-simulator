from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import fastf1
import pandas as pd

def get_completed_rounds():
    """Fetches the CURRENT year schedule and returns all rounds that have finished."""
    fastf1.Cache.enable_cache('/opt/airflow/cache')
    
    # DYNAMIC YEAR: Automatically gets the current year
    current_year = datetime.now().year
    schedule = fastf1.get_event_schedule(current_year)
    
    # 1. Get today's date with a buffer
    today = datetime.now() - timedelta(days=1)
    
    # 2. FILTER LOGIC:
    # We remove the 'conventional' restriction to include Sprint weekends.
    # We only care that the 'EventDate' (the Sunday Race) is in the past.
    completed = schedule[schedule['EventDate'] < today]
    
    # We also filter out 'Testing' sessions which aren't real races
    completed = completed[completed['EventFormat'] != 'testing']
    
    rounds = completed['RoundNumber'].tolist()
    print(f"Found {len(rounds)} completed rounds for {current_year} to process: {rounds}")
    return rounds

with DAG(
    'f1_season_master_orchestrator',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@weekly', 
    catchup=False,
    tags=['master']
) as dag:

    check_schedule = PythonOperator(
        task_id='get_completed_rounds',
        python_callable=get_completed_rounds
    )

    def trigger_ingestions(**kwargs):
        ti = kwargs['ti']
        rounds = ti.xcom_pull(task_ids='get_completed_rounds')
        
        current_year = datetime.now().year

        for r_num in rounds:
            # FIX: Ensure trigger_dag_id matches 'f1_ingestion_pipeline_v3'
            TriggerDagRunOperator(
                task_id=f'trigger_round_{r_num}',
                trigger_dag_id='f1_strategy_production_pipeline', 
                conf={'year': current_year, 'round': r_num},
                wait_for_completion=False
            ).execute(kwargs)

    run_all_ingestions = PythonOperator(
        task_id='trigger_all_completed_rounds',
        python_callable=trigger_ingestions
    )

    check_schedule >> run_all_ingestions