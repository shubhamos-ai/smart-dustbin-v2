import torch
import requests
import os
import cv2
import time
import json
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime

# ---------------- CONFIG ----------------
IP_FILE = "ip.txt"
PORT = 8000
CONFIDENCE_THRESHOLD = 0.60
CAPTURE_DELAY = 2  # seconds after detection
DETECTION_LOG = "detection.json"

# ---------------- LOAD MODEL (OFFLINE MODE) ----------------
print("Loading CLIP model from local cache...")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Try loading with local_files_only first
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True, use_fast=True)
    print("✓ Model loaded successfully in OFFLINE mode!")
except Exception as e:
    print(f"⚠️ Could not load model in strict offline mode. Error: {e}")
    print("Attempting standard load (may require internet if not cached)...")
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    print("✓ Model loaded successfully!")

model.eval()

# ---------------- LABELS & CATEGORIZATION ----------------
labels = [
    "a photo of a plastic bottle",
    "a photo of biodegradable waste",
    "a photo of metal waste",
    "a photo of glass waste",
    "a photo of paper waste",
    "a photo of food waste",
    "a photo of cardboard waste",
    "a photo of organic waste",
    "a photo of electronic waste",
    "a photo of battery waste"
]

# Waste categorization
ORGANIC_WASTE = ["biodegradable_waste", "food_waste", "organic_waste", "paper_waste", "cardboard_waste"]
INORGANIC_WASTE = ["plastic_bottle", "metal_waste", "glass_waste", "electronic_waste", "battery_waste"]

endpoint_map = {
    labels[0]: "plastic_bottle",
    labels[1]: "biodegradable_waste",
    labels[2]: "metal_waste",
    labels[3]: "glass_waste",
    labels[4]: "paper_waste",
    labels[5]: "food_waste",
    labels[6]: "cardboard_waste",
    labels[7]: "organic_waste",
    labels[8]: "electronic_waste",
    labels[9]: "battery_waste",
}

def get_waste_category(waste_type):
    """Determine if waste is organic or inorganic"""
    if waste_type in ORGANIC_WASTE:
        return "organic"
    elif waste_type in INORGANIC_WASTE:
        return "inorganic"
    return "unknown"

# ---------------- FASTAPI ----------------
app = FastAPI(title="EcoLogic - Smart Waste Management")
app.mount("/author", StaticFiles(directory="author"), name="author")

def load_ip():
    try:
        with open(IP_FILE, "r") as f:
            return f.read().strip()
    except:
        return "localhost"

def save_ip(ip):
    with open(IP_FILE, "w") as f:
        f.write(ip)

def save_detection(waste_type, category, confidence):
    """Save detection to log file"""
    try:
        if os.path.exists(DETECTION_LOG):
            with open(DETECTION_LOG, "r") as f:
                data = json.load(f)
        else:
            data = {"detections": []}
        
        data["detections"].append({
            "waste_type": waste_type,
            "category": category,
            "confidence": round(confidence * 100, 2),
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 detections
        data["detections"] = data["detections"][-50:]
        
        with open(DETECTION_LOG, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving detection: {e}")

def classify_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    pred_index = probs.argmax().item()
    confidence = probs[0][pred_index].item()

    return endpoint_map[labels[pred_index]], confidence

def send_ping(category):
    """Send ping to ESP32 with category (organic/inorganic)"""
    ip = load_ip()
    url = f"http://{ip}:{PORT}/{category}"
    try:
        requests.get(url, timeout=3)
        print(f"✓ Sent → {url}")
    except Exception as e:
        print(f"✗ Ping failed: {e}")

# ---------------- CAMERA SELECTION ----------------
CAMERA_INDEX_FILE = "camera_index.txt"

def load_camera_index():
    try:
        if os.path.exists(CAMERA_INDEX_FILE):
            with open(CAMERA_INDEX_FILE, "r") as f:
                return int(f.read().strip())
    except:
        pass
    return 0

def save_camera_index(index):
    with open(CAMERA_INDEX_FILE, "w") as f:
        f.write(str(index))

def get_camera_selection_at_startup():
    import glob
    print("\n" + "="*30)
    print("      CAMERA SELECTION")
    print("="*30)
    
    video_devices = glob.glob("/dev/video*")
    video_devices.sort()
    
    if not video_devices:
        print("⚠️ No cameras found in /dev/video*. Defaulting to index 0.")
        return 0
    
    print("Available Cameras:")
    for dev in video_devices:
        try:
            idx = int(dev.replace("/dev/video", ""))
            print(f"  [{idx}] {dev}")
        except:
            continue
    
    try:
        # Check if we already have a saved index
        saved_idx = load_camera_index()
        print(f"\nUsing saved camera index: {saved_idx} (from {CAMERA_INDEX_FILE})")
        return saved_idx
    except:
        return 0

selected_camera_index = get_camera_selection_at_startup()
camera = cv2.VideoCapture(selected_camera_index)

if not camera.isOpened():
    print(f"⚠️ Warning: Could not open camera {selected_camera_index}. It might be busy or invalid.")
else:
    print(f"✅ Camera {selected_camera_index} initialized successfully!\n")

def switch_camera(new_index):
    global camera, selected_camera_index
    print(f"🔄 Switching camera to index {new_index}...")
    
    # Release old camera
    if camera is not None:
        camera.release()
    
    # Initialize new camera
    new_camera = cv2.VideoCapture(new_index)
    if new_camera.isOpened():
        camera = new_camera
        selected_camera_index = new_index
        save_camera_index(new_index)
        print(f"✅ Successfully switched to camera {new_index}")
        return True
    else:
        print(f"❌ Failed to open camera {new_index}, reverting to old camera if possible.")
        # Try to revert to old camera
        camera = cv2.VideoCapture(selected_camera_index)
        return False
current_detection = {"type": "Waiting for trigger...", "category": "", "confidence": 0, "timestamp": ""}
detection_in_progress = False
latest_frame = None

def generate_frames():
    """Continuously capture frames for live feed"""
    global latest_frame
    
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
            
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        # Store latest frame for detection
        latest_frame = frame.copy()

        # Show prediction text on video
        cv2.putText(frame, "EcoLogic - Smart Detection", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def process_detection():
    """Process detection when triggered by /object endpoint"""
    global current_detection, detection_in_progress, latest_frame
    
    if detection_in_progress:
        print("⚠️ Detection already in progress, skipping...")
        return {"status": "busy", "message": "Detection already in progress"}
    
    if latest_frame is None:
        print("⚠️ No frame available")
        return {"status": "error", "message": "No camera frame available"}
    
    detection_in_progress = True
    
    try:
        # Step 1: Object trigger received
        print("� Object detection triggered! Waiting 2 seconds to capture...")
        current_detection = {
            "type": "Analyzing...",
            "category": "PROCESSING",
            "confidence": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # Step 2: Wait 2 seconds
        time.sleep(CAPTURE_DELAY)
        
        # Step 3: Capture frame and send to CLIP model
        print("🤖 Sending to CLIP model for classification...")
        frame_to_analyze = latest_frame.copy()
        waste_type, confidence = classify_frame(frame_to_analyze)
        category = get_waste_category(waste_type)
        
        # Step 4: Check if something was actually detected
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"❌ Nothing detected (confidence: {round(confidence*100,2)}%)")
            current_detection = {
                "type": "Nothing detected",
                "category": "",
                "confidence": round(confidence * 100, 2),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            detection_in_progress = False
            return {"status": "nothing", "confidence": round(confidence * 100, 2)}
        
        # Step 5: Display result
        print(f"✅ Classification complete!")
        print(f"📊 Detected: {waste_type} → {category.upper()} ({round(confidence*100,2)}%)")
        
        # Step 6: Update UI with result
        current_detection = {
            "type": waste_type.replace("_", " ").title(),
            "category": category.upper(),
            "confidence": round(confidence * 100, 2),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # Step 7: Call ESP32 once
        print(f"📡 Calling ESP32 endpoint: /{category}")
        send_ping(category)
        save_detection(waste_type, category, confidence)
        print(f"✓ ESP32 notified successfully!")
        
        # Step 8: Reset for next detection
        detection_in_progress = False
        print("🔄 Ready for next detection!\n")
        
        return {
            "status": "success",
            "waste_type": waste_type,
            "category": category,
            "confidence": round(confidence * 100, 2)
        }
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        detection_in_progress = False
        current_detection = {
            "type": "Error occurred",
            "category": "",
            "confidence": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        return {"status": "error", "message": str(e)}

# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/detection_status")
def detection_status():
    """API endpoint for current detection status"""
    return current_detection

@app.get("/shubhamos", response_class=HTMLResponse)
async def config_page():
    """Configuration page for IP and Camera"""
    current_ip = load_ip()
    current_camera = selected_camera_index
    
    # List available cameras for the UI
    import glob
    video_devices = glob.glob("/dev/video*")
    video_devices.sort()
    cameras_list = []
    for dev in video_devices:
        try:
            idx = int(dev.replace("/dev/video", ""))
            cameras_list.append({"index": idx, "path": dev})
        except:
            continue

    with open("templates/config.html", "r") as f:
        html = f.read()
    
    # Basic templating
    html = html.replace("{{CURRENT_IP}}", current_ip)
    html = html.replace("{{CURRENT_CAMERA}}", str(current_camera))
    
    # Generate camera options HTML
    camera_options = ""
    for cam in cameras_list:
        selected = "selected" if cam["index"] == current_camera else ""
        camera_options += f'<option value="{cam["index"]}" {selected}>Camera {cam["index"]} ({cam["path"]})</option>'
    
    html = html.replace("{{CAMERA_OPTIONS}}", camera_options)
    
    return html

@app.post("/shubhamos/update")
async def update_settings(ip: str = Form(None), camera_index: int = Form(None)):
    """Update settings (IP and/or Camera)"""
    success_msg = ""
    
    if ip is not None:
        save_ip(ip)
        success_msg += "ip=1&"
        
    if camera_index is not None:
        if camera_index != selected_camera_index:
            if switch_camera(camera_index):
                success_msg += "cam=1&"
            else:
                success_msg += "cam_err=1&"
    
    return RedirectResponse(url=f"/shubhamos?{success_msg}success=1", status_code=303)

@app.get("/object")
async def trigger_detection():
    """Trigger object detection - called by external system when object is placed"""
    print("\n" + "="*50)
    print("🎯 /object endpoint triggered!")
    print("="*50)
    result = process_detection()
    return result


@app.get("/organic")
async def organic_endpoint():
    """Endpoint for organic waste"""
    return {"status": "organic bin activated"}

@app.get("/inorganic")
async def inorganic_endpoint():
    """Endpoint for inorganic waste"""
    return {"status": "inorganic bin activated"}
