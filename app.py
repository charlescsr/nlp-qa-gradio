import os
import sys
from transformers import pipeline
import gradio as gr

model = pipeline('question-answering', model='deepset/tinyroberta-squad2', tokenizer='deepset/tinyroberta-squad2')

def qa(passage, question):
    question = question
    context = passage
    nlp_input = {
        'question': question,
        'context': context
    }

    return model(nlp_input)['answer']

passage = "The quick brown fox jumped over the lazy dog."
question = "Who jumps over the lazy dog?"

iface = gr.Interface(qa,
                        title="Question Answering using RoBERTa",
                        inputs=[gr.inputs.Textbox(lines=15), "text"],
                        outputs=["text"],
                        examples=[["{}".format(passage), "{}".format(question)]])
iface.launch()