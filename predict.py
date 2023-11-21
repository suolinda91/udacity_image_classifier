import argparse
import utils

parser = argparse.ArgumentParser(description='Predict flower name from an image')

parser.add_argument('image_path', help='Path to image')
parser.add_argument('saved_model', help='Path to pretrained model')
parser.add_argument('--top_k', default=5, type=int, help='Return the top K most likely classes')
parser.add_argument('---category_names', default='label_map.json', help='Path to a JSON file mapping labels to flower names')

args = parser.parse_args()
top_k = 1 if args.top_k < 1 else args.top_k
category_names = args.category_names
model = utils.load_keras_model(args.saved_model)
probs, class_indices = utils.predict(args.image_path, model, top_k)

class_names = utils.get_class_names(args.category_names, class_indices)

print(probs)
print(class_names)