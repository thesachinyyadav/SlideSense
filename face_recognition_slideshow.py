import cv2
import face_recognition
import numpy as np
import os
import time
from threading import Thread

# Load student datasets
student_categories = ['science', 'arts', 'commerce']
student_encodings = {category: [] for category in student_categories}
student_images = {category: [] for category in student_categories}

print("Loading student datasets...")
# Load known student images
for category in student_categories:
    path = f"students/{category}/"
    try:
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            print(f"Loading {img_path}")
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                student_encodings[category].append(encodings[0])
                student_images[category].append(filename)
        print(f"Loaded {len(student_encodings[category])} {category} student images")
    except FileNotFoundError:
        print(f"Warning: Directory not found: {path}")

# Load posters
poster_images = {}
print("\nLoading poster images...")
for category in student_categories:
    path = f"posters/{category}/"
    try:
        poster_images[category] = [os.path.join(path, f) for f in os.listdir(path)]
        print(f"Loaded {len(poster_images[category])} {category} posters")
    except FileNotFoundError:
        print(f"Warning: Directory not found: {path}")

# Variables for slideshow
current_slideshow_category = None
slideshow_running = False
stop_slideshow = False

def recognize_students(frame):
    """Recognize students from the camera feed and count their categories."""
    # Convert to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    
    # Find faces in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    
    # Initialize category count
    category_count = {category: 0 for category in student_categories}
    recognized_faces = []
    
    # Process each detected face
    for i, (face_encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
        # Scale back face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        matched_category = None
        
        # Check against each category
        for category in student_categories:
            if student_encodings[category]:  # Check if there are encodings for this category
                matches = face_recognition.compare_faces(student_encodings[category], face_encoding, tolerance=0.6)
                if True in matches:
                    category_count[category] += 1
                    matched_category = category
                    break
        
        # Draw rectangle with color based on category
        if matched_category == "science":
            color = (0, 255, 0)  # Green for science
        elif matched_category == "arts":
            color = (0, 0, 255)  # Red for arts
        elif matched_category == "commerce":
            color = (255, 0, 0)  # Blue for commerce
        else:
            color = (200, 200, 200)  # Gray for unknown
            
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Add label
        label = f"{matched_category.capitalize() if matched_category else 'Unknown'} #{i+1}"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        recognized_faces.append({
            "location": (top, right, bottom, left),
            "category": matched_category
        })
    
    # Determine majority category
    majority_category = max(category_count, key=category_count.get) if sum(category_count.values()) > 0 else None
    
    # Add text showing detection counts
    y_pos = 30
    cv2.putText(frame, f"Detected Students:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_pos += 30
    for category in student_categories:
        color = (0, 255, 0) if category == "science" else (0, 0, 255) if category == "arts" else (255, 0, 0)
        cv2.putText(frame, f"{category.capitalize()}: {category_count[category]}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 25
    
    if majority_category:
        cv2.putText(frame, f"MAJORITY: {majority_category.upper()}", (10, y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame, majority_category, category_count

def text_slideshow(category):
    """Run a text-based slideshow in the console for the given category."""
    global stop_slideshow, slideshow_running
    
    if not category in poster_images or not poster_images[category]:
        print(f"No posters available for {category.capitalize()}")
        slideshow_running = False
        return
    
    try:
        slideshow_running = True
        print(f"\n----- STARTING {category.upper()} SLIDESHOW -----")
        
        posters = poster_images[category]
        slide_index = 0
        
        while not stop_slideshow:
            # Clear console (cross-platform)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            poster_path = posters[slide_index]
            poster_name = os.path.basename(poster_path)
            
            print(f"\n{'=' * 50}")
            print(f"  {category.upper()} DEPARTMENT - POSTER {slide_index + 1}/{len(posters)}")
            print(f"  {poster_name}")
            print(f"{'=' * 50}")
            print(f"\nFile path: {poster_path}")
            print("\n(Press 'q' to quit slideshow, 's' to stop recognition)")
            
            # Move to next slide
            slide_index = (slide_index + 1) % len(posters)
            
            # Wait for 3 seconds before showing next slide
            time.sleep(3)
        
        print("\n----- SLIDESHOW ENDED -----\n")
        slideshow_running = False
        
    except Exception as e:
        print(f"Error in slideshow: {e}")
        slideshow_running = False

def main():
    global current_slideshow_category, slideshow_running, stop_slideshow
    
    print("\n===== STUDENT RECOGNITION SYSTEM =====")
    print("Press 'q' to quit")
    print("Press 's' to stop/restart recognition")
    print("Press 'p' to pause/resume\n")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize variables
    category_history = []  # Store recent detections to ensure stability
    paused = False
    system_active = True
    
    try:
        while system_active:
            if not paused:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, majority_category, category_count = recognize_students(frame)
                
                # Update category history for stability
                if majority_category:
                    category_history.append(majority_category)
                    if len(category_history) > 15:  # Keep last 15 frames
                        category_history.pop(0)
                
                # Check for stable majority (same category for 10 out of 15 frames)
                stable_majority = None
                if len(category_history) >= 10:
                    most_common = max(set(category_history), key=category_history.count)
                    if category_history.count(most_common) >= 10:
                        stable_majority = most_common
                        
                        # Start slideshow in a separate thread if category changed and no slideshow is running
                        if stable_majority != current_slideshow_category and not slideshow_running:
                            current_slideshow_category = stable_majority
                            stop_slideshow = False
                            
                            print(f"\nStable majority detected: {stable_majority.upper()} students")
                            slideshow_thread = Thread(target=text_slideshow, args=(stable_majority,))
                            slideshow_thread.daemon = True
                            slideshow_thread.start()
                
                # Show active category in the display
                if slideshow_running and current_slideshow_category:
                    cv2.putText(processed_frame, f"SLIDESHOW: {current_slideshow_category.upper()}", 
                               (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the processed frame
                cv2.imshow("Student Recognition", processed_frame)
            
            # Check for key presses (with short timeout to keep responsive)
            key = cv2.waitKey(1) & 0xFF
            
            # Process key commands
            if key == ord('q'):  # Quit
                system_active = False
                stop_slideshow = True
                print("Exiting program...")
            elif key == ord('s'):  # Stop/restart recognition
                if slideshow_running:
                    stop_slideshow = True
                    print("Stopping slideshow...")
                else:
                    # Reset recognition state
                    category_history = []
                    current_slideshow_category = None
                    print("Recognition reset.")
            elif key == ord('p'):  # Pause/resume
                paused = not paused
                print("Recognition", "paused" if paused else "resumed")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Program ended")

if __name__ == "__main__":
    main()