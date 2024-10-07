import gradio as gr
import argparse
from main import run_benchmark
from utils.data import load_json
import os

# Load configurations
model_config = load_json('configs/models.json')['models']
dataset_config = load_json('configs/datasets.json')['datasets']
prompt_config = load_json('configs/prompts.json')

# Extract options from configurations
model_options = [model['model_dest'] for model in model_config.values()]
dataset_options = list(dataset_config.keys())
prompt_options = list(set([prompt for prompts in prompt_config.values() for prompt in prompts.keys()]))

def run_benchmark_ui(models, datasets, replications, test_modes, temperatures, max_task_tokens, prompts, suffix, refine, seed):
    # Convert inputs to the format expected by run_benchmark
    models = models.split(',')
    datasets = datasets.split(',')
    test_modes = test_modes.split(',')
    temperatures = [float(t) for t in temperatures.split(',')]
    max_task_tokens = [int(t) for t in max_task_tokens.split(',')]
    prompts = prompts.split(',') if prompts else None
    refine = [refine]

    # Get the active conda environment
    server = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')

    # Run the benchmark
    predictions_df, agg_metrics = run_benchmark(
        server, models, datasets, replications, test_modes,
        temperatures, max_task_tokens, prompts, suffix, refine, seed
    )

    # Generate a summary of the results
    summary = f"Benchmark completed.\n"
    summary += f"Total runs: {len(agg_metrics)}\n"
    summary += f"Average accuracy: {agg_metrics['accuracy'].mean():.2f}\n"
    summary += f"Results saved in: results/{suffix}\n"

    return summary

# Create the Gradio interface
iface = gr.Interface(
    fn=run_benchmark_ui,
    inputs=[
        gr.Dropdown(choices=model_options, multiselect=True, label="Models"),
        gr.Dropdown(choices=dataset_options, multiselect=True, label="Datasets"),
        gr.Slider(minimum=1, maximum=100, step=1, default=10, label="Replications"),
        gr.Dropdown(choices=["unconstrained", "constrained"], multiselect=True, default=["unconstrained"], label="Test Modes"),
        gr.Textbox(default="0.75", label="Temperatures (comma-separated)"),
        gr.Textbox(default="2500", label="Max Task Tokens (comma-separated)"),
        gr.Dropdown(choices=prompt_options, multiselect=True, label="Prompts (optional)"),
        gr.Textbox(default="", label="Suffix"),
        gr.Checkbox(default=False, label="Refine Predictions"),
        gr.Checkbox(default=True, label="Set Seed"),
    ],
    outputs="text",
    title="LLM Benchmark Framework",
    description="Select options and run the benchmark",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)