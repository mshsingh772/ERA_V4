# Image Processor

A simple web application built with Flask that allows users to upload an image and apply various filters to it. The application displays both the original and the processed image side-by-side.

## Features

*   **Image Upload:** Easily upload images in PNG, JPG, JPEG, and GIF formats.
*   **Multiple Filters:** Apply a variety of image processing filters:
    *   No Filter (original image)
    *   Grayscale
    *   Gaussian Blur
    *   Find Edges (overall edge detection)
    *   Horizontal Edges (custom kernel for horizontal edge detection)
    *   Vertical Edges (custom kernel for vertical edge detection)
*   **Real-time Processing:** Images are processed and displayed dynamically upon filter selection.
*   **Session Management:** Retains the uploaded image across filter selections using Flask sessions.
*   **Responsive Display:** Simple HTML/CSS for a user-friendly interface.

## Setup and Installation

Follow these steps to get the application up and running on your local machine.

### Prerequisites

*   Python 3.x
*   pip (Python package installer)

### Installation

1.  **Navigate to the project directory:**
    If you're in the root `ERA_V4` directory, move into the `Session2` folder:
    ```bash
    cd Session2
    ```

2.  **Install dependencies:**
    The application requires Flask for the web server and Pillow (PIL) for image processing.
    ```bash
    pip install Flask Pillow
    ```

### Running the Application

1.  **Start the Flask development server:**
    From inside the `Session2` directory, run the `app.py` file:
    ```bash
    python app.py
    ```
2.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000/` (or `localhost:5000`).

## Usage

1.  **Upload an Image:**
    *   On the homepage, you will see an "Upload Image" form.
    *   Click "Choose File" and select an image from your computer.
    *   Click "Upload & Process".
2.  **Apply Filters:**
    *   Once an image is uploaded, it will be displayed along with a dropdown menu for "Select Filter".
    *   Choose a filter from the dropdown (e.g., "Grayscale", "Gaussian Blur", "Horizontal Edges").
    *   The processed image will automatically update on the right side.
3.  **Upload New Image:**
    *   To upload a different image, click the "Upload New Image" button. This will clear the current session and allow you to upload a new file.
