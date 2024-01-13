import torch
from PIL import Image


from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

import cv2
import argparse


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def sample_image(source):
    # Examples of how to use the function
    # display_media("//path/to/image.jpg")  # for image
    # display_media("path/to/video.mp4")  # for video file
    # display_media(0)  # for webcam (typically 0 for the default camera)
    # display_media("rtsp://username:password@ip_address:port/path")  # for CCTV or IP camera

    if isinstance(source, str) and source.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):  # image file
        image = cv2.imread(source)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image_resized = cv2.resize(image_rgb, (596, 437))
            # cv2.imshow('Image', image_resized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("Image not found or unable to open.")

        return image_rgb
        
        
            
    else:  # Video or live stream
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Failed to open the video source.")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB, resize and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_resized = cv2.resize(frame_rgb, (596, 437))
            # cv2.imshow('Video', frame_resized)

            # if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            #     break

        # cap.release()
        # cv2.destroyAllWindows()
        return frame_rgb


# caption = "South-east asian riding a motorcycle not protective gear in a highway"
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
def load_model():
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    return model, vis_processors, text_processors

def inference(raw_image, caption, model, vis_processors, text_processors) -> str:

    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    score = f'{itm_scores[:, 1].item():.2%}'
    # print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
    return score

def main():
    parser = argparse.ArgumentParser(description="Display media using OpenCV")
    parser.add_argument('source', help="The source of the media. It can be a path to an image or video, webcam index, or video stream URL.")
    args = parser.parse_args()
    model, vis_processors, text_processors = load_model()
    caption = input("Caption: ")
    raw_image = Image.fromarray(sample_image(args.source))
    print(type(raw_image))
    print(raw_image)
    score:str = inference(raw_image=raw_image, caption=caption, model=model, vis_processors=vis_processors, text_processors=text_processors)
    print(f"Score: {score}")


if __name__=="__main__":
    main()