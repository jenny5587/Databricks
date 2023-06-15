# Databricks notebook source
# MAGIC %md
# MAGIC ###webhook

# COMMAND ----------

# MAGIC %pip install databricks-registry-webhooks

# COMMAND ----------

ls

# COMMAND ----------

# MAGIC %run ./API_Helpers

# COMMAND ----------

access_token = 'your_token'
model_name = 'spark_mlops'
job_id = get_job_id() 
slack_url = "https://hooks.slack.com/services/Txxxxxxxxxx/Bxxxxxxxxxx/Oxxxxxxxxxxxxx"

# COMMAND ----------

from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec, HttpUrlSpec
# Create a HTTP webhook that will create alerts about registered models created
http_url_spec = HttpUrlSpec(url=slack_url, secret="secret_string")
http_webhook = RegistryWebhooksClient().create_webhook(
  events=["TRANSITION_REQUEST_CREATED", "MODEL_VERSION_CREATED", "MODEL_VERSION_TRANSITIONED_STAGE"],
  http_url_spec=http_url_spec,
  model_name=model_name
)
http_webhook

# COMMAND ----------

# Test the HTTP webhook
# RegistryWebhooksClient().test_webhook(id=http_webhook.id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manage Webhooks

# COMMAND ----------

list_webhooks("spark_mlops")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ![image](https://github.com/jenny5587/TIL/assets/103649749/44c0509b-5253-48f0-b652-dd5faebd3e0c)
