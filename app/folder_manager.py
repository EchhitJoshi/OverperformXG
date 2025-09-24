import yaml
import os 


config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

output_path = ""
submodel_name = ""
encoding_path = ""
feature_report_path = ""
llm_code_path = config['HOME_DIRECTORY'] + "/app/llm_code_output.py"




