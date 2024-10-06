from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
import io

from datetime import datetime, timedelta
import pandas as pd
import json
import logging

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get('S3_BUCKET')
DEFAULT_ARGS = {
    'owner': 'larchenkov-mihail',
    'retries': 3,
    'retry_delay': timedelta(minutes=1)
}

def init(ti, model_name):
    timestamp = datetime.now().isoformat()
    
    ti.xcom_push(key='init_metrics', value={'timestamp': timestamp, 'model_name': model_name})

def get_data(ti, model_name):
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
        key=f'{DEFAULT_ARGS["owner"]}/{model_name}/datasets/dataset.csv',
        bucket_name=BUCKET,
        replace=True
    )

def prepare_data(ti, model_name):
    start_time = datetime.now().isoformat()

    s3_hook = S3Hook('s3_connection')
    csv_buffer = io.BytesIO()
    s3_hook.get_key(key=f'{DEFAULT_ARGS["owner"]}/{model_name}/datasets/dataset.csv', bucket_name=BUCKET).download_fileobj(csv_buffer)
    csv_buffer.seek(0)
    
    df = pd.read_csv(csv_buffer)
    _LOG.info('Dataframe has been uploaded successfully!')

    X = df.drop('MedHouseVal', axis=1)
    _LOG.info(f'X shape is: {X.shape}')
    y = df['MedHouseVal']
    _LOG.info(f'y shape is: {y.shape}')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _LOG.info('X has been scaled successfully!')
    cols = list(X.columns)
    
    end_time = datetime.now().isoformat()

    prepared_buffer = io.BytesIO()
    pd.DataFrame(X_scaled, columns=cols).to_csv(prepared_buffer, index=False)
    s3_hook.load_bytes(
        bytes_data=prepared_buffer.getvalue(),
        key=f'{DEFAULT_ARGS["owner"]}/{model_name}/datasets/prepared_data.csv',
        bucket_name=BUCKET,
        replace=True
    )

    ti.xcom_push(key='prepare_data_metrics', value={'start_time': start_time, 'end_time': end_time, 'features': cols})

def train_model(ti, model_name):
    start_time = datetime.now().isoformat()

    s3_hook = S3Hook('s3_connection')
    prepared_buffer = io.BytesIO()
    s3_hook.get_key(key=f'{DEFAULT_ARGS["owner"]}/{model_name}/datasets/prepared_data.csv', bucket_name=BUCKET).download_fileobj(prepared_buffer)
    prepared_buffer.seek(0)
    
    X = pd.read_csv(prepared_buffer)
    _LOG.info(f'X shape is: {X.shape}')
    y = fetch_california_housing(as_frame=True).frame['MedHouseVal']
    _LOG.info(f'y shape is: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    _LOG.info('X and y are divided into train and test. Test size - 25%')

    if model_name == 'linear_regression':
        model = LinearRegression()
    elif model_name == 'decision_tree':
        model = DecisionTreeRegressor()
    elif model_name == 'random_forest':
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    _LOG.info('Model has been fitted successfully!')
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    _LOG.info(f'{model_name} MSE = {mse:.4f}')
    
    end_time = datetime.now().isoformat()

    ti.xcom_push(key='train_model_metrics', value={'start_time': start_time, 'end_time': end_time, 'mse': mse})

def save_results(ti, model_name):
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
        key=f'{DEFAULT_ARGS["owner"]}/{model_name}/results/results.json',
        bucket_name=BUCKET,
        replace=True
    )

for model_name in ['linear_regression', 'decision_tree', 'random_forest']:
    with DAG(
        dag_id=f'{DEFAULT_ARGS["owner"]}_{model_name}',
        description=f'DAG - {model_name}',
        schedule_interval='0 1 * * *',
        start_date=days_ago(30),
        catchup=False,
        tags=['mlops'],
        default_args=DEFAULT_ARGS
    ) as dag:
        
        init_task = PythonOperator(task_id='init', python_callable=init, op_kwargs={'model_name': model_name})
        get_data_task = PythonOperator(task_id='get_data', python_callable=get_data, op_kwargs={'model_name': model_name})
        prepare_data_task = PythonOperator(task_id='prepare_data', python_callable=prepare_data, op_kwargs={'model_name': model_name})
        train_model_task = PythonOperator(task_id='train_model', python_callable=train_model, op_kwargs={'model_name': model_name})
        save_results_task = PythonOperator(task_id='save_results', python_callable=save_results, op_kwargs={'model_name': model_name})

        init_task >> get_data_task >> prepare_data_task >> train_model_task >> save_results_task
