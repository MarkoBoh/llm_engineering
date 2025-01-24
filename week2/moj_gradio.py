import gradio as gr

def velike_crke(text):
    return text.upper()

gr.interface.Interface(fn=velike_crke, inputs="text", outputs="textbox", flagging_mode= "never").launch(share=True)

