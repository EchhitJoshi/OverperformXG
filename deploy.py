import os
import shutil
import argparse

def get_latest_model(directory):
    """Finds the latest model directory based on creation time."""
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return None
    
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not subdirs:
        return None
        
    return max(subdirs, key=os.path.getctime)

def deploy_model(model_folder_name=None):
    """
    Deploys a model from `outputs/models` to `deployed_models`.
    
    If `model_folder_name` is not provided, it deploys the latest model.
    """
    source_base_dir = "outputs/models"
    deployment_base_dir = "deployed_models"

    if model_folder_name:
        source_dir = os.path.join(source_base_dir, model_folder_name)
        if not os.path.exists(source_dir):
            print(f"Error: Model folder '{model_folder_name}' not found in '{source_base_dir}'.")
            return
    else:
        print("No model name provided, finding the latest model...")
        latest_model_dir = get_latest_model(source_base_dir)
        if not latest_model_dir:
            print(f"No models found in '{source_base_dir}'.")
            return
        source_dir = latest_model_dir
        model_folder_name = os.path.basename(latest_model_dir)
        print(f"Found latest model: '{model_folder_name}'")

    # Clear the deployment directory
    if os.path.exists(deployment_base_dir):
        print(f"Clearing the '{deployment_base_dir}' directory...")
        for item in os.listdir(deployment_base_dir):
            item_path = os.path.join(deployment_base_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
                return

    # Create deployment directory if it doesn't exist
    os.makedirs(deployment_base_dir, exist_ok=True)
    
    # Define the destination directory
    destination_dir = os.path.join(deployment_base_dir, model_folder_name)

    # Copy the model folder
    try:
        print(f"Copying '{source_dir}' to '{destination_dir}'...")
        shutil.copytree(source_dir, destination_dir)
        print("Deployment successful!")
    except Exception as e:
        print(f"An error occurred during deployment: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a model from 'outputs/models' to 'deployed_models'.")
    parser.add_argument(
        "model_name",
        nargs='?',
        default=None,
        help="Optional: The name of the model folder to deploy. If not provided, the latest model will be deployed."
    )
    args = parser.parse_args()
    
    deploy_model(args.model_name)
