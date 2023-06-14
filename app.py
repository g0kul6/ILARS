import cv2
import gradio as gr 
from app_eval import eval


with gr.Blocks() as demo:
    video_upload = gr.UploadButton(label="Upload the Video", file_types=["video"])
    slider = gr.Slider(maximum=200,interactive=True,steps=1)
    frames = []
    def get_frame(video):
        frames.clear()
        cap = cv2.VideoCapture(video.name)
        i = 0
        for i in range(201):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if ret == False:
                break
            frames.append(frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()
    video_upload.upload(fn=get_frame, inputs=[video_upload])
    def return_frame(index):
        img = frames[index]
        return img
    slider.change(return_frame,slider,gr.Image(shape=(1280, 720),type="numpy"))    
    question = gr.Textbox(label="Question")
    model_type = gr.CheckboxGroup(["SurgGPT"],label="Model Choice")
    # model_version = gr.CheckboxGroup(["efvlegpt2Swin","efvlegpt2ViT","efvlegpt2rs18"],label="Model Version")
    # model_version = "efvlegpt2rs18"
    # model_type = gr.CheckboxGroup(["LCGN","VisualBERT"],label="Model Choice")
    # model_version = gr.CheckboxGroup(["vb","vb-rr","lcgn","lcgn-rr"],label="Model Version")
    answer = gr.Textbox(label="Answer")
    description = gr.Textbox(label="Description")
    predict = gr.Button(value="Predict")
    def predict_ans(index,question,model_choice):
        img = frames[index]
        ans,description = eval(model_ver="efvlegpt2rs18",img=img,question=question)
        # ans = "hi"
        return ans,description
    predict.click(fn=predict_ans,inputs=[slider,question,model_type],outputs=[answer,description])
    
if __name__ == "__main__":
    demo.launch()
