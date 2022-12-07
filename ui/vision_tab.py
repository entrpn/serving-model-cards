import gradio as gr

def vision_tab(sa_file_loc, deploy):
    with gr.Blocks() as image_tab:
        # with gr.TabItem("Vision", id='vision_tab'):
        with gr.Column():
            image_dropdown = gr.Dropdown(label='Choose a model', choices=['image-classification', 'dreambooth'], interactive=True)
            image_model_description = gr.Markdown('')
            gr.Markdown("Prepare your data")
        with gr.Column(visible=False, label='dreambooth_upload') as dreambooth_upload:
            gr.Markdown("You can use one of the two methods below.")
            gr.Radio(['Upload folder','Use .CSV'])
            gr.Markdown("TODO")
            gr.Markdown("You can use one of the two methods below.")
        with gr.Column(visible=False, ) as image_class_upload:
            gr.Markdown("You can use one of the two methods below.")
            gr.Radio(['Upload folder','Use .CSV'])
            gr.Markdown("You can use one of the two methods below.")
        with gr.Row():
            data_upload = gr.File(Interactive=True)
        with gr.Row():
            deploy_img_btn = gr.Button("deploy")
    def update_on_img_dropdown(input_choice):
        retval = ''
        if input_choice == 'image-classification':
            retval = 'Classify images'
        elif input_choice == 'dreambooth':
            retval = 'Personalize text2image models like stable diffusion with a few images of a subject.'
            #dreambooth_upload_widget.render()
        return {
            image_model_description : retval,
            image_class_upload : gr.update(visible=True) if input_choice == 'image-classification' else gr.update(visible=False),
            dreambooth_upload : gr.update(visible=True) if input_choice == 'dreambooth' else gr.update(visible=False)
        }
    image_dropdown.change(update_on_img_dropdown, inputs=[image_dropdown], outputs=[image_model_description, image_class_upload, dreambooth_upload])
    deploy_img_btn.click(fn=deploy, inputs=[sa_file_loc,image_dropdown],outputs=[])
