import yaml
with open("config.yaml","r") as file:
    config = yaml.safe_load(file)

output_path = ""
submodel_name = ""
encoding_path = ""
feature_report_path = ""
llm_code_path = config['HOME_DIRECTORY'] + "/app/llm_code_output.py"




