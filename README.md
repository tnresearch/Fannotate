![Banner](bin/banner1.png)
# Fannotate
 Faster annotation with Fannotate - this repo is the beginning of the UI which will be used to annotate text training data _faster_ with the use of LLMs.

# Why?
 - Currently, its not realistic or resposible to use LLMs in production for high-volume text classification tasks, as this can be solved much faster and cheaper with targeted text classfiers trained on the distribution of the task at hand. 
- Rather, using the LLM to generate training data will result in cost savings both at inference time, as well as when annotating the training data.

# How?
- This framework therefore help the ML team create annotations of trainig data in a responsible manner, by:
    - Providing initial quality assessments
    - Enabling the user to adapt the 'codebook' or description of each label over time

From here, the documentation of the HUGINN benchmark framework is preserved in order to make the dataption more smooth. The HUGINN codebase is provided in ``src/fannotate/`` at the time of writing (07/10/2024).


# HUGINN benchmark
This is a dedicated framework aimed at testing ICL-capabilities in LLMs within professional services (process-related specialized knowledge).

## Benchmark environments and modes

- Full-precision (fp16) using **Transformers**:
    - **Chat**: Using the raw output from the model (with regex)
    - **Constrained**: Using outlines and FSM's to constrain the output to the target labels (classificati only)
- Quantization with **llama.cpp**:
    - **Chat**: Using the raw output from the model (with regex)

# Docker
## Installing
- Clone this repo to a local folde
- Create new folder ``data/`` and ``models/``
- Add datasets (instructions to be added here ASAP) into folder: ``data/``
- Add GGUF files if any to ``models/``
- Update configuration files in the `configs/` directory:
    - `models.json` for any models added
    - `datasets.json` for any datasets added
    - `prompts.json` for any prompts added
- Build the docker image:
    - ``docker build -t huginn .``

## Running a benchmark
- Navigate to cloned HUGINN directory containing ``data/`` folder with tasks
- ``mkdir results``
- Run docker image:
    - **Llama.cpp**:
      ``docker run -e HF_TOKEN="<your_token>" --gpus 1 --name huginn --rm -v "$(pwd)":/workspace -it huginn llama.cpp python main.py --models models/Meta-Llama-3-8B-Instruct.Q2_K.gguf --datasets Lydmaskinen --replications 1 --test_modes unconstrained --temperatures 0.0 --max_task_tokens 100 --refine False --seed True --suffix Experiment_name``
    - **Transformers**:
      ``docker run -e HF_TOKEN="<your_token>" --gpus 1 --name huginn --rm -v "$(pwd)":/workspace -it huginn transformers python main.py --models google/gemma-2-2b-it --datasets Lydmaskinen --replications 1 --test_modes unconstrained --temperatures 0.0 --max_task_tokens 100 --refine False --seed True --suffix Experiment_name``


## Arguments:

- `--models`: List of model names to evaluate (must match `model_dest` in `models.json`)
- `--datasets`: List of dataset names to use (must match `dataset_name` in `datasets.json`)
- `--replications`: Number of replications (default: 10)
- `--test_modes`: List of test modes (e.g., "constrained", "unconstrained", default: ["unconstrained"])
- `--temperatures`: List of temperature values (default: [0.75])
- `--max_task_tokens`: List of maximum task token values (default: [2500])
- `--prompts`: List of prompts to use (default: [all prompts associated with the dataset])
- `--seed`: Whether to use deterministic sampling (True or False)
- `--suffix`: String to add to the foldername for the results (useful when running multiple tests in a row)




## Output

The framework will create a new directory in the `results/` folder for each run, containing:

- `design_table.csv`: The experimental design table
- `emissions.csv`: The emissions produced during the experiment
- `predictions.h5`: Raw prediction results on task(row)-level
- `agg_metrics.h5`: Aggregated performance metrics
- Copy of the configuration files

The notebook `latest_results.ipynb` will load the latest experiment and show the results.
|
## Notes

- Ensure you have the necessary API keys and model files as required by your chosen server type.
- The framework uses CodeCarbon to track emissions during benchmarking.
- Incompatible model-prompt pairs and datasets will be automatically filtered out.

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.
