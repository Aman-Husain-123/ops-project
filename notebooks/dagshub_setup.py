import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/Aman-Husain-123/ops-project.mlflow')
dagshub.init(repo_owner='Aman-Husain-123', repo_name='ops-project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)