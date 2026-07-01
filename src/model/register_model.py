# register model

import json
import mlflow
import logging
import os
import dagshub

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

dagshub_url = "https://dagshub.com"
repo_owner = "Aman-Husain-123"
repo_name = "ops-project"

# Set up MLflow tracking URI with basic auth
mlflow.set_tracking_uri(f'https://{repo_owner}:{dagshub_token}@dagshub.com/{repo_owner}/{repo_name}.mlflow')


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        client = mlflow.tracking.MlflowClient()

        # Get the artifact URI for the run and build the model source path
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        model_source = f"{artifact_uri}/{model_path}"
        logger.debug(f'Model source URI: {model_source}')

        # Ensure the registered model exists
        try:
            client.create_registered_model(model_name)
            logger.debug(f'Created new registered model: {model_name}')
        except Exception:
            logger.debug(f'Registered model {model_name} already exists.')

        # Create a new model version using the artifact source
        model_version = client.create_model_version(
            name=model_name,
            source=model_source,
            run_id=run_id
        )

        # Transition the model to "Staging" stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
