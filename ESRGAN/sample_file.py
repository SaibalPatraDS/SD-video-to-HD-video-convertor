import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = r"models\RRDB_ESRGAN_x4.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def upsample_video(input_video_path, output_video_path, model_path=model_path, device=device, frame_skip=2):
    # Load Pretrained Model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    # Open the Input Video File
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("File Path Not Defined")
        return
    
    # Get Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # CREATE VIDEO WRITER OBJECT
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 720))

    print(f"Processing Video: {input_video_path}")
    print(f"Frame Size: {width}X{height}, FPS: {fps}, Total Frames: {total_frames}")

    # PROCESS EACH FRAME
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # END of Video

        # Skip frames if frame_count is not divisible by frame_skip
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # UPSCALE the frame
        img = frame * 1.0 / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        # Resize the output to 1280*720
        output = cv2.resize(output, (1280, 720), interpolation=cv2.INTER_LINEAR_EXACT)

        # Writing the final images onto the video
        out.write(output)

        frame_count += 1

        print(f"\rProcessed Frame: {frame_count}/{total_frames}", end="")

    print("\nProcessing Completed")

    # Release Video Capture and Video Writer Object
    cap.release()
    out.release()

# Using the function with frame_skip set to 2 (processing every other frame)
upsample_video(input_video_path=r"Videoes\Input\balcony_sample.mp4",
               output_video_path=r"Videos\Output\sample_output.mp4",
               device=device,
               model_path=model_path,
               frame_skip=2)
