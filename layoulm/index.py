from preprocess import *
import pytesseract

from torch.nn import CrossEntropyLoss

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

labels = get_labels("./data/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index


from PIL import Image, ImageDraw
image = Image.open('./test1.jpg')
image = image.convert("RGB")

model_path='./layoutlm.pt'
model=model_load(model_path,num_labels)
image, words, boxes, actual_boxes = preprocess("./test1.jpg")

word_level_predictions, final_boxes=convert_to_features(image, words, boxes, actual_boxes, model)
print(word_level_predictions, final_boxes)
draw = ImageDraw.Draw(image, "RGBA")

def iob_to_label(label):
  if label != 'O':
    return label[2:]
  else:
    return ""
label2color = {'question':'blue', 'answer':'green', 'header':'black', '':'red'}
for prediction, box in zip(word_level_predictions, final_boxes):
    predicted_label = iob_to_label(label_map[prediction]).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
#     draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label])

image.save('testing1.jpg')
