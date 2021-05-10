from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace, Datastore, Experiment, Dataset, Environment


ws = Workspace.from_config()

# Define Compute Target
compute_name = "automate-311"
compute_target = ws.compute_targets[compute_name]

# Configure Environment
aml_run_config = RunConfiguration()
aml_run_config.target = compute_target
aml_run_config.environment.python.user_managed_dependencies = False
aml_run_config.environment.name = 'automate_311_env'
aml_run_config.environment.version = '1.0'

# Add some packages relied on by data prep step
aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
    conda_packages=['scikit-learn'], 
    pip_packages=['azure-storage-blob', 'python-Levenshtein', 'gensim', 'nltk'], 
    pin_sdk_version=False)

input_audio_sas_url = PipelineParameter(name="input_audio_sas_url", default_value="")
output_transcription_sas_url = PipelineParameter(name="output_transcription_sas_url", default_value="")

step = PythonScriptStep(name="Complete 311 Automation Workflow",
                        script_name="main_workflow.py",
                        arguments=["--inputAudioSASUrl", input_audio_sas_url, "--outputTranscriptionSASUrl", output_transcription_sas_url],
                        compute_target=compute_target,
                        source_directory='./')

# Build the pipeline
pipeline1 = Pipeline(workspace=ws, steps=[step])

# # Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'automate_311_env').submit(pipeline1)

# PUBLISH HERE

# ADD PARAMS

# pipeline_run1.wait_for_completion()
