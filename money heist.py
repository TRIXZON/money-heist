from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from collections import Counter
from webcolors import rgb_to_name, CSS3_HEX_TO_NAMES, hex_to_rgb
import openai
import os
import hashlib

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = "sk-svcacct-yKD3gzcyEzFGz9K8V3LehYdunwe_j1gIjDS33VaWZh7NEVonQnJic5a4-oVxKT3BlbkFJrgcNZBH9Emye0Lmkbzh11yxTtSkIwj30TfVoj6oEwVGPAmYOAkuxpj4Cu4cAA"

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Define HTML element descriptions
html_element_descriptions = {
    "button": "A clickable button, often used for submitting forms or triggering actions.",
    "input": "A text input field for the user to enter data, such as name or email.",
    "link": "A clickable link that navigates to another page.",
    "image": "An image displayed on the webpage.",
    "navbar": "A navigation bar containing links to important sections of the site.",
}

# Function to find the closest color name
def closest_color(requested_color):
    min_colors = {}
    for hex_code, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = hex_to_rgb(hex_code)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# Function to extract the dominant color
def get_dominant_color(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    pixels = image.getdata()
    most_common_color = Counter(pixels).most_common(1)[0][0]
    try:
        color_name = rgb_to_name(most_common_color)
    except ValueError:
        color_name = closest_color(most_common_color)
    return color_name

# Function to generate HTML and CSS
def generate_html_css(description, color):
    prompt = f"""
    Create an HTML and CSS snippet based on the following details:
    - Element: {description}
    - Color: {color}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful web development assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content']

# Function to hash images for caching
def hash_image(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Cache to avoid duplicate processing
processed_images = {}

# Route for processing image uploads
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_path = os.path.join('temp', image.filename)
    image.save(image_path)

    # Check if the image has already been processed
    image_hash = hash_image(image_path)
    if image_hash in processed_images:
        return jsonify(processed_images[image_hash])

    img = Image.open(image_path)
    description, color = get_description_from_image(img)
    generated_code = generate_html_css(description, color)

    result = {
        'description': description,
        'color': color,
        'generated_code': generated_code
    }

    # Cache the result
    processed_images[image_hash] = result

    return jsonify(result)

def get_description_from_image(image):
    inputs = processor(images=image, text=list(html_element_descriptions.values()), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.matmul(image_features, text_features.T)

    best_match_idx = similarity.argmax().item()
    best_match_label = list(html_element_descriptions.keys())[best_match_idx]
    description = html_element_descriptions[best_match_label]

    dominant_color = get_dominant_color(image)
    return description, dominant_color

# Route to serve the index.html
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)  # Temporary folder for image uploads
    app.run(debug=True)
