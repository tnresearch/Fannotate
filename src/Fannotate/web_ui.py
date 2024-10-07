import gradio as gr
import os
import argparse
from main import main
#from main import main as run_benchmark
from utils.data import load_json

# Load configurations
model_config = load_json('configs/models.json')['models']
dataset_config = load_json('configs/datasets.json')['datasets']
prompt_config = load_json('configs/prompts.json')

# Extract options from configurations
model_options = [model['model_dest'] for model in model_config.values()]
dataset_options = list(dataset_config.keys())

def get_prompts_for_dataset(dataset):
    return list(prompt_config.get(dataset, {}).keys())

def update_prompts(dataset):
    return gr.Dropdown(choices=get_prompts_for_dataset(dataset))

def run_benchmark_ui(models, dataset, prompts, replications, test_modes, temperatures, max_task_tokens, suffix, refine, seed):
    # Convert inputs to the format expected by run_benchmark
    models = models if isinstance(models, list) else [models]
    datasets = [dataset]  # We're only allowing one dataset at a time
    test_modes = test_modes if isinstance(test_modes, list) else [test_modes]
    temperatures = [float(t) for t in temperatures.split(',')]
    max_task_tokens = [int(t) for t in max_task_tokens.split(',')]
    prompts = prompts if isinstance(prompts, list) else [prompts] if prompts else None
    refine = [refine]

    # Create an argparse.Namespace object to mimic CLI arguments
    args = argparse.Namespace(
        models=models,
        datasets=datasets,
        replications=replications,
        test_modes=test_modes,
        temperatures=temperatures,
        max_task_tokens=max_task_tokens,
        prompts=prompts,
        suffix=suffix,
        refine=refine,
        seed=seed
    )

    # Run the benchmark
    result = main(args)

    return f"Benchmark completed. Results saved in: results/ folder"

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Fannotate LLM Benchmark UI")
    with gr.Row():
        with gr.Column():
            models = gr.Dropdown(choices=model_options, multiselect=True, label="Models")
            dataset = gr.Dropdown(choices=dataset_options, label="Dataset")
            prompts = gr.Dropdown(multiselect=True, label="Prompts")
            replications = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Replications")
            test_modes = gr.Dropdown(choices=["unconstrained", "constrained"], multiselect=True, value=["unconstrained"], label="Test Modes")
        with gr.Column():
            temperatures = gr.Textbox(value="0.0", label="Temperatures (comma-separated)")
            max_task_tokens = gr.Textbox(value="2500", label="Max Task Tokens (comma-separated)")
            suffix = gr.Textbox(value="", label="Suffix")
            refine = gr.Checkbox(value=False, label="Refine Predictions")
            seed = gr.Checkbox(value=True, label="Set Seed")
    
    run_button = gr.Button("Run Benchmark")
    output = gr.Textbox(label="Output")

    # Set up interactivity
    dataset.change(update_prompts, inputs=[dataset], outputs=[prompts])
    run_button.click(
        run_benchmark_ui,
        inputs=[models, dataset, prompts, replications, test_modes, temperatures, max_task_tokens, suffix, refine, seed],
        outputs=[output]
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)