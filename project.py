from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

from datetime import datetime, timedelta
import io
import os
import pandas as pd
import json
import logging

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get('S3_BUCKET')
DEFAULT_ARGS = {
    "owner": 'larchenkov-mihail',
    'retries': 3,
    'retry_delay': timedelta(minutes=1)
}

def configure_mlflow():
    for key in [
        'MLFLOW_TRACKING_URI',
        'AWS_ENDPOINT_URL',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_DEFAULT_REGION',
    ]:
        os.environ[key] = Variable.get(key)

def init(ti):
    configure_mlflow()
    
    experiment_name = 'larchenkov-mihail'
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(name=experiment_name)
        _LOG.info(f'Experiment {experiment_name} created with ID: {experiment_id}')
    else:
        experiment_id = experiment.experiment_id
        _LOG.info(f'Experiment {experiment_name} already exists with ID: {experiment_id}')

    with mlflow.start_run(run_name='garotar', experiment_id=experiment_id, description='parent') as parent_run:
        parent_run_id = parent_run.info.run_id
        _LOG.info(f'Created parent run with ID: {parent_run_id}')

        metrics = {
            'experiment_id': experiment_id,
            'parent_run_id': parent_run_id,
            'timestamp': datetime.now().isoformat()
        }
        ti.xcom_push(key='init_metrics', value=metrics)


def get_data(ti):
    start_time = datetime.now().isoformat()
    
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    _LOG.info(df.columns)
    
    end_time = datetime.now().isoformat()
    
    ti.xcom_push(key='get_data_metrics', value={'start_time': start_time, 'end_time': end_time, 'dataset_size': df.shape})

    s3_hook = S3Hook('s3_connection')
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    s3_hook.load_bytes(
        bytes_data=csv_buffer.getvalue(),
        key=f'{DEFAULT_ARGS["owner"]}/project/datasets/dataset.csv',
        bucket_name=BUCKET,
        replace=True
    )

def prepare_data(ti):
    start_time = datetime.now().isoformat()

    s3_hook = S3Hook('s3_connection')
    csv_buffer = io.BytesIO()
    s3_hook.get_key(key=f'{DEFAULT_ARGS["owner"]}/project/datasets/dataset.csv', bucket_name=BUCKET).download_fileobj(csv_buffer)
    csv_buffer.seek(0)
    
    df = pd.read_csv(csv_buffer)
    _LOG.info('Dataframe has been uploaded successfully!')

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    _LOG.info('X has been scaled successfully!')
    cols = list(X.columns)
    
    end_time = datetime.now().isoformat()

    prepared_buffer = io.BytesIO()
    pd.DataFrame(X_scaled, columns=cols).to_csv(prepared_buffer, index=False)
    s3_hook.load_bytes(
        bytes_data=prepared_buffer.getvalue(),
        key=f'{DEFAULT_ARGS["owner"]}/project/datasets/prepared_data.csv',
        bucket_name=BUCKET,
        replace=True
    )

    ti.xcom_push(key='prepare_data_metrics', value={'start_time': start_time, 'end_time': end_time, 'features': cols})

def train_model(ti, model_name):
    configure_mlflow()

    init_metrics = ti.xcom_pull(key='init_metrics')
    experiment_id = init_metrics['experiment_id']
    parent_run_id = init_metrics['parent_run_id']

    _LOG.info(f'experiment_id: {experiment_id}')
    _LOG.info(f'parent_run_id: {parent_run_id}')

    s3_hook = S3Hook('s3_connection')
    prepared_buffer = io.BytesIO()
    s3_hook.get_key(key=f'{DEFAULT_ARGS["owner"]}/project/datasets/prepared_data.csv', bucket_name=BUCKET).download_fileobj(prepared_buffer)
    prepared_buffer.seek(0)

    X = pd.read_csv(prepared_buffer)
    y = fetch_california_housing(as_frame=True).frame['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    models = {
        'linear_regression': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor()
    }

    model = models[model_name]

    _LOG.info(f'Attempting to start a nested run for model {model_name} under parent run {parent_run_id}')
    
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:
            _LOG.info(f'Started nested run for model {model_name} with run ID: {child_run.info.run_id}')

            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            
            mse = mean_squared_error(y_test, prediction)
            r2 = r2_score(y_test, prediction)

            mlflow.log_metric('MSE', mse)
            _LOG.info(f'MSE = {mse}')
            mlflow.log_metric('R2', r2)
            _LOG.info(f'R2 = {r2}')

            eval_df = pd.DataFrame(X_val).copy()
            eval_df['target'] = y_val

            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)

            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets='target',
                model_type='regressor',
                evaluators=['default'],
            )

            _LOG.info(f'Model {model_name} has been fitted and logged successfully!')

    ti.xcom_push(key='train_model_metrics', value={'model_name': model_name})
    
def save_results(ti):
    s3_hook = S3Hook('s3_connection')

    init_metrics = ti.xcom_pull(key='init_metrics')
    get_data_metrics = ti.xcom_pull(key='get_data_metrics')
    prepare_data_metrics = ti.xcom_pull(key='prepare_data_metrics')
    train_model_metrics = ti.xcom_pull(key='train_model_metrics')

    results = {
        'init_metrics': init_metrics,
        'get_data_metrics': get_data_metrics,
        'prepare_data_metrics': prepare_data_metrics,
        'train_model_metrics': train_model_metrics
    }

    results_buffer = io.BytesIO()
    results_buffer.write(json.dumps(results).encode())
    s3_hook.load_bytes(
        bytes_data=results_buffer.getvalue(),
        key=f'{DEFAULT_ARGS["owner"]}/project/results/results.json',
        bucket_name=BUCKET,
        replace=True
    )

with DAG(
    dag_id='larchenkov-mihail',
    schedule_interval='0 1 * * *',
    start_date=days_ago(30),
    catchup=False,
    tags=['mlops'],
    default_args=DEFAULT_ARGS
) as dag:
    
    init_task = PythonOperator(task_id='init', python_callable=init)
    get_data_task = PythonOperator(task_id='get_data', python_callable=get_data)
    prepare_data_task = PythonOperator(task_id='prepare_data', python_callable=prepare_data)
    train_linear_regression_task = PythonOperator(task_id='train_linear_regression', python_callable=train_model, op_kwargs={'model_name': 'linear_regression'})
    train_decision_tree_task = PythonOperator(task_id='train_decision_tree', python_callable=train_model, op_kwargs={'model_name': 'decision_tree'})
    train_random_forest_task = PythonOperator(task_id='train_random_forest', python_callable=train_model, op_kwargs={'model_name': 'random_forest'})
    save_results_task = PythonOperator(task_id='save_results', python_callable=save_results)

    init_task >> get_data_task >> prepare_data_task >> [train_linear_regression_task, train_decision_tree_task, train_random_forest_task] >> save_results_task