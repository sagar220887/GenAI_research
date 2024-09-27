import os
import pandas as pd
from pandasai import SmartDataframe
from langchain.llms import CTransformers


# step 1 : Load dataset
parent_dir = os.path.dirname(os.getcwd())
df = pd.read_csv(os.path.join(parent_dir, 'data', 'census.csv'))

# step 2: Initialize LLM
# model = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./model/ --local-dir-use-symlinks False

model_path = os.path.join(os.path.dirname(parent_dir), 'model', 'pytorch_model.bin')

llm=CTransformers(
            model=model_path,
            model_type="llama",
            config={'max_new_tokens':128,
                    'temperature':0.01}
    )

# step 3: Create a SmartDataframe
smart_df = SmartDataframe(df, config={"llm": llm})
