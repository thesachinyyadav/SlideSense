# SlideSense - Face Recognition Slideshow System

**Current Version: 1.0**

A smart face recognition system that automatically detects students from different academic streams and displays relevant posters based on the majority audience present.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Overview

SlideSense uses real-time face recognition to identify students belonging to different academic categories (Science, Arts, Commerce) and automatically triggers a slideshow of relevant posters when a stable majority of students from a particular stream is detected.

## Features

- **Real-time Face Recognition**: Uses webcam feed to detect and recognize faces
- **Multi-category Classification**: Supports Science, Arts, and Commerce student categories
- **Automatic Slideshow Trigger**: Starts relevant slideshow when 10+ detections of same category occur in 15 frames
- **Visual Feedback**: 
  - Color-coded bounding boxes (Green=Science, Red=Arts, Blue=Commerce)
  - Live count display of detected students per category
  - Majority indicator on screen
- **Interactive Controls**: Pause, reset, or quit the system with keyboard shortcuts

## Project Structure

```
SlideSense/
├── face_recognition_slideshow.py    # Main application
├── students/                        # Student face images for training
│   ├── science/                     # Science students
│   ├── arts/                        # Arts students
│   └── commerce/                    # Commerce students
├── posters/                         # Slideshow posters
│   ├── science/                     # Science-related posters
│   ├── arts/                        # Arts-related posters
│   └── commerce/                    # Commerce-related posters
└── README.md
```

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- face_recognition
- numpy

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/thesachinyyadav/SlideSense.git
   cd SlideSense
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python face_recognition numpy
   ```

   > **Note**: The `face_recognition` library requires `dlib`. On Windows, you may need to install Visual Studio Build Tools first.

3. **Add student images**
   - Place student face images in the appropriate category folders under `students/`
   - Images should be clear, front-facing photos with good lighting
   - Supported formats: JPG, PNG, JPEG

4. **Add posters**
   - Place poster images in the appropriate category folders under `posters/`
   - These will be displayed in the slideshow when that category is detected

## Usage

Run the main script:

```bash
python face_recognition_slideshow.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Stop current slideshow / Reset recognition |
| `p` | Pause / Resume recognition |

## How It Works

1. **Initialization**: Loads all student face images and creates face encodings for each category
2. **Detection**: Captures video frames and detects faces using face_recognition library
3. **Recognition**: Compares detected faces against stored encodings (tolerance: 0.6)
4. **Category Tracking**: Maintains a history of the last 15 frames to determine stable majority
5. **Slideshow Trigger**: When 10+ detections in 15 frames belong to same category, starts slideshow
6. **Display**: Shows posters from the majority category in a 3-second rotation

## Adding New Students

1. Take a clear photo of the student's face
2. Save it in the appropriate category folder:
   - `students/science/student_name.jpg`
   - `students/arts/student_name.jpg`
   - `students/commerce/student_name.jpg`
3. Restart the application to load new faces

## Adding New Posters

1. Add poster images to the relevant category folder under `posters/`
2. Restart the application to include new posters in the slideshow

## Configuration

Key parameters can be adjusted in the code:

- **Face matching tolerance**: `tolerance=0.6` (lower = stricter matching)
- **Frame resize factor**: `0.25` (smaller = faster processing, less accurate)
- **Stability threshold**: 10 detections out of 15 frames
- **Slideshow interval**: 3 seconds per poster

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not opening | Check if camera is connected and not used by another app |
| Faces not detected | Ensure good lighting and clear face visibility |
| Wrong category matches | Add more training images for better accuracy |
| Slow performance | Reduce frame resolution or increase resize factor |

## License

This project is open source and available under the MIT License.

## Author

Sachin Yadav

---

*SlideSense - Smart audience-aware presentation system*
