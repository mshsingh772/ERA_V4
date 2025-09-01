# session2/app.py
import os
import uuid
import time # Import time for cache busting
from flask import Flask, request, render_template, send_from_directory, session, redirect, url_for
from PIL import Image, ImageFilter

app = Flask(__name__)
# Set a secret key for session management. CHANGE THIS IN PRODUCTION!
app.secret_key = 'your_super_secret_key_here'

# Define upload and processed folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Custom kernels for edge detection
HORIZONTAL_EDGE_KERNEL = ImageFilter.Kernel((3, 3), [
    -1, -1, -1,
     0,  0,  0,
     1,  1,  1
], scale=1)

VERTICAL_EDGE_KERNEL = ImageFilter.Kernel((3, 3), [
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
], scale=1)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(original_img_path, filter_type):
    """
    Applies the selected filter to the image.
    Returns the processed PIL Image object.
    """
    img = Image.open(original_img_path)

    if filter_type == 'grayscale':
        processed_img = img.convert('L')
    elif filter_type == 'gaussian':
        processed_img = img.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_type == 'edges':
        processed_img = img.filter(ImageFilter.FIND_EDGES)
    elif filter_type == 'horizontal_edges':
        processed_img = img.filter(HORIZONTAL_EDGE_KERNEL)
    elif filter_type == 'vertical_edges':
        processed_img = img.filter(VERTICAL_EDGE_KERNEL)
    else: # No filter or 'none' selected
        processed_img = img.copy() # Just copy the image
    return processed_img

@app.route('/', methods=['GET'])
def index():
    original_image_filename = session.get('original_image_filename')
    processed_image_filename = None
    selected_filter = request.args.get('filter', 'none')
    cache_buster = int(time.time()) # For browser cache busting

    if original_image_filename:
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_image_filename)

        if os.path.exists(original_filepath):
            # Process the image with the selected filter
            processed_img = process_image(original_filepath, selected_filter)

            # Always save to a fixed name, e.g., 'processed_image.png'
            # We'll use the original image's extension for the processed image if possible
            processed_image_ext = os.path.splitext(original_image_filename)[1]
            processed_image_filename = 'processed_image' + processed_image_ext
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_filename)
            processed_img.save(processed_filepath)
        else:
            # Original image not found, clear session and redirect
            session.pop('original_image_filename', None)
            return redirect(url_for('index'))

    return render_template('index.html',
                           original_image=original_image_filename,
                           processed_image=processed_image_filename,
                           selected_filter=selected_filter,
                           cache_buster=cache_buster)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique filename for the original image
        original_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_filepath)

        # Store the unique filename in the session
        session['original_image_filename'] = original_filename

        # Redirect to the main page to display and allow filter selection
        return redirect(url_for('index'))
    else:
        # If file type is not allowed, render index with an error message
        return render_template('index.html', message='Allowed image types are png, jpg, jpeg, gif')

@app.route('/reset', methods=['POST'])
def reset_image():
    # Clear the original image from the session
    session.pop('original_image_filename', None)
    # Optionally, remove the processed image file
    processed_image_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_image.png')
    if os.path.exists(processed_image_filepath):
        os.remove(processed_image_filepath)
    # Redirect to the main page to show the upload form
    return redirect(url_for('index'))


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
