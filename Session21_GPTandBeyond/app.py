import numpy as np
import pandas as pd
import gradio as gr
from model import *

device = 'cpu'
model = GPTLanguageModel()
model = model.to(device)
model.load_state_dict(torch.load("tiny_gpt.pth", map_location=torch.device('cpu')))

def generate_weird_tale(context = None, max_new_tokens = 500):
    if context is None:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.unsqueeze(torch.tensor(encode(context), dtype=torch.long, device=device),0)
    #print(context.shape)
    wtale = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
    
    return wtale

title = "Shakespeare's Weird Tales"
description = '''It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.'''
examples = [[None,500],["Hi",1000]]

context = gr.TextArea(value=None, label="Do you want to give a context ?")
max_new_tokens = gr.Slider(1, 10000, value = 500, step=1, label="How long should the tale be (in characters)?")
wtale = gr.TextArea(label="Your weird tale")

inps = [
      context,
      max_new_tokens
  ]

output = [
  wtale
]

demo = gr.Interface(
    generate_weird_tale, 
    inputs = inps, 
    outputs = output,
    title = title,
    description = description,
    examples = examples,
)
demo.launch()