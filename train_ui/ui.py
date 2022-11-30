import gradio as gr
# from image_upload_renders import dreambooth_upload
from vision_tab import vision_tab
from view_creations import view_creations

from css_and_js import js

# dreambooth_upload_widget = dreambooth_upload()
# dreambooth_upload_widget.update(visibility=False)

def deploy(sa_file_loc, model_type):
    print("sa_file_loc:",sa_file_loc)
    print("model_type:",model_type)

with gr.Blocks() as demo:
    with gr.Tabs() as MainMenuTabs:
        with gr.TabItem('View', id='view_tab'):
            view_creations()
        with gr.TabItem('Train', id='train_tab'):
            gr.Markdown("# Train UI")
            with gr.Column():
                with gr.Accordion("## Project setup"):
                    gr.Markdown("""Train UI helps users easily train OSS models with little or no code using Vertex AI pipelines.
                    All you need is a [GCP project](https://cloud.google.com/).
                    """)
                    gr.Markdown('## Project setup')
                    sa_file_loc = gr.File(file_count=1, Interactive=True)
                    gr.Markdown("""There are two ways to authenticateYou'll need a service account or an authenticated user with the following permissions:
                    - Vertex AI
                    - GCS storage admin
                    """)
                    gr.Markdown('Note: If this machine is already authenticated with GCP, this is not needed.')
            with gr.Row():
                with gr.Column():
                    gr.Markdown('## Select the model type')
                    with gr.Tabs() as tabs:
                        with gr.TabItem("Vision", id='vision_tab'):
                            vision_tab(sa_file_loc, deploy)
                        with gr.TabItem("Text", id='nlp_tab'):
                            with gr.Column():
                                nlp_dropdown = gr.Dropdown(label='Choose a model', choices=['text-classification', 'question answering'], interactive=True)
                                nlp_model_description = gr.Markdown('', label='model_description')
                            with gr.Row():
                                deploy_nlp_btn = gr.Button("deploy")
    # deploy_btn = gr.Button("deploy",)

    demo.load(_js=js())

demo.launch(share=False)