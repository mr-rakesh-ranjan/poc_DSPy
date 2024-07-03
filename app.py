
#  setting DSPy

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dotenv import load_dotenv, find_dotenv
import os 
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

# set up the LLM
turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct',api_key=api_key , max_tokens=250)
dspy.settings.configure(lm=turbo)

# load math questions from datasets
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

# print(gsm8k_trainset)

from dspy.teleprompt import BootstrapFewShot
from ChainOfThought import CoT

config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

teleprompter =  BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
evaluate(optimized_cot)

turbo.inspect_history(n=1)

