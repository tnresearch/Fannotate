# Running the LLM Benchmark Web UI with Conda
To run the web UI through Conda, you'll need to create a new Conda environment and install the necessary dependencies. Here's a step-by-step guide with markup code describing how to set up and run the web UI:

## 1. Create a new Conda environment

```bash
conda create -n llm_benchmark python=3.8
conda activate llm_benchmark
```

## 2. Install dependencies

Create a `requirements.txt` file with the following content:

```
numpy==1.26.4
pandas
scikit-learn
outlines
openpyxl
protobuf
transformers
accelerate
sentencepiece
datasets
codecarbon
matplotlib
seaborn
h5py
gradio
tqdm
psutil
requests
torch
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## 3. Set up the project structure

Ensure your project structure looks like this:

```
llm_benchmark/
├── configs/
│   ├── datasets.json
│   ├── models.json
│   └── prompts.json
├── utils/
│   ├── data.py
│   ├── model_factory.py
│   └── prompt_factory.py
├── main.py
├── inference.py
├── web_ui.py
└── requirements.txt
```

## 4. Run the Web UI

Execute the following command in your terminal:

```bash
python web_ui.py
```

## 5. Access the Web UI

Open a web browser and navigate to:

```
http://localhost:7860
```

You should now see the LLM Benchmark Web UI. Use the interface to select models, datasets, and other parameters, then click the "Run Benchmark" button to start the evaluation.