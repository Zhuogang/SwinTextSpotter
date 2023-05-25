import torch
import torchvision
import torch.neuron
import cv2
import numpy as np
device = torch.device("cuda")
model = torch.jit.load('model_traced_480_cuda.pt', map_location=device)

# torch.rand(size=(3,640,1000))

# print(model(torch.rand(size=(3,1000,1000))))
# print(list(model.parameters())[0].shape, list(model.parameters())[1].shape, list(model.parameters())[2].shape )

# print(model)

nW = 480
nH = 480

model.eval()
input_img_path = "test_img/392287007.png"
img_orig = cv2.imread(input_img_path)
img_orig = cv2.resize(img_orig, [320, 320], interpolation = cv2.INTER_CUBIC)


with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
    # Apply pre-processing to image.

    original_image = img_orig[:, :, ::-1]
    
    height, width = original_image.shape[:2]
        
    resized_image = cv2.resize(original_image, [nW, nH], interpolation = cv2.INTER_CUBIC)

    image = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1)).to(device)
    
    predictions = model(image)
    
m, n = predictions[0].size()
results = torch.cat((predictions[0], predictions[3].view(m, 1)), 1)
results = results[results[:, 4] > 0.2]

# instances = predictions["instances"]
# instances = instances[instances.scores > confidence_threshold]
outputs = results[:, :4].detach().cpu().numpy()
print(outputs.shape)

save_path = "results_ts_output_2.jpg"
def plot(image, boxes, save_path, scale=nW/320):
    #image in the form of cv2.imread()
    # gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    N, _ = boxes.shape
    for i in range(N):
        color = [100, 50, 100]
        # print(box)
        cv2.rectangle(image, (int(boxes[i][0]/scale), int(boxes[i][1]/scale)), (int(boxes[i][2]/scale), int(boxes[i][3]/scale)), color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path, image)

plot(img_orig, outputs, save_path)