import gradio as gr

def dreambooth_upload():
    with gr.Blocks() as retval:
        with gr.Row(visible=False):
            gr.Markdown("Prepare your data")
            gr.Markdown("You can use one of the two methods below.")
            gr.Radio(['Upload folder','Use .CSV'])
    return retval