import dagshub
dagshub.init(repo_owner='kunal957662', repo_name='mlflow_expriment', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
