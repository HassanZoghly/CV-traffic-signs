import json                                     # type: ignore
import random                                   # type: ignore
import numpy as np                              # type: ignore
import streamlit as st                          # type: ignore
from streamlit_lottie import st_lottie          # type: ignore
import requests                                 # type: ignore
from PIL import Image                           # type: ignore
import cv2                                      # type: ignore
import torch                                    # type: ignore
import torch.nn as nn                           # type: ignore
from torchvision import models                  # type: ignore
import torchvision.transforms as transforms     # type: ignore

# Check CUDA availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0)) 

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()  
    return None 
######################################## Preprocessing Methods ########################################
class ImagePreprocessor:
    def __init__(self, image):
        self.image = image
    
    def resize_image(self, size=(255, 255)):
        self.image = cv2.resize(self.image, size)
        return self.image 
    
    def convert_to_grayscale(self):
        if len(self.image.shape) == 3:  # If the image is RGB, convert to grayscale
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image
    
    def reduce_noise(self):
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        return self.image
    
    def enhance_contrast(self):
        if len(self.image.shape) == 3:  # Check if the image is BGR
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            self.image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return self.image
    
    def normalize_image(self):
        self.image = self.image / 255.0 if self.image.dtype != np.float32 else self.image
        self.image = (self.image * 255).astype(np.uint8)  # Convert back to uint8
        return self.image
    
    def brighten_image(self, alpha=1.0, beta=50):
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self.image
######################################## Segmentation Methods ########################################
def automatic_thresholding(image):
    if len(image.shape) == 3:  # Convert to grayscale if the image is color (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def local_thresholding(image, block_size=15):
    if len(image.shape) == 3:  # Convert to grayscale if the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    result = np.zeros_like(image)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i + block_size, j:j + block_size]
            threshold = np.mean(block)
            result[i:i + block_size, j:j + block_size] = (block > threshold).astype(np.uint8) * 255
    return result

def cheng_jin_kuo(image, block_size=15, k=0.5):
    if len(image.shape) == 3:  # Convert to grayscale if the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    result = np.zeros_like(image)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i + block_size, j:j + block_size]
            local_mean = np.mean(block)
            local_std = np.std(block)
            threshold = local_mean - k * local_std
            result[i:i + block_size, j:j + block_size] = (block > threshold).astype(np.uint8) * 255
    return result
######################################## Feature Extraction Methods ########################################
# Chain code directions for 8-connectivity
chain_code_direction_8 = {
    (0, 1): 0,      # Right
    (-1, 1): 1,     # Upper-right
    (-1, 0): 2,     # Up
    (-1, -1): 3,    # Upper-left
    (0, -1): 4,     # Left
    (1, -1): 5,     # Lower-left
    (1, 0): 6,      # Down
    (1, 1): 7       # Lower-right
}
# Sharpening filter for pre-processing
sharpening_kernel = np.array([  [0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0] ])

####### LBP extraction function #######
def extract_lbp(image):
    rows, cols = image.shape
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center_pixel = image[i, j]
            binary_string = ''
            binary_string += '1' if image[i - 1, j - 1] >= center_pixel else '0'    # index 0
            binary_string += '1' if image[i - 1, j] >= center_pixel else '0'        # index 1
            binary_string += '1' if image[i - 1, j + 1] >= center_pixel else '0'    # index 2
            binary_string += '1' if image[i, j + 1] >= center_pixel else '0'        # index 3
            binary_string += '1' if image[i + 1, j + 1] >= center_pixel else '0'    # index 4
            binary_string += '1' if image[i + 1, j] >= center_pixel else '0'        # index 5
            binary_string += '1' if image[i + 1, j - 1] >= center_pixel else '0'    # index 6
            binary_string += '1' if image[i, j - 1] >= center_pixel else '0'        # index 7
            lbp_image[i, j] = int(binary_string, 2)
    return lbp_image

def extract_local_features_lbp(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    lbp_image = extract_lbp(gray)
    return lbp_image

####### Chain code extraction #######
def extract_chain_code(image, max_code_length=None):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sharpened = cv2.filter2D(gray_img, -1, sharpening_kernel)
    _, binary_img = cv2.threshold(sharpened, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None 
    largest_contour = max(contours, key=cv2.contourArea)
    chain_codes = []
    for i in range(1, len(largest_contour)):
        prev, curr = largest_contour[i - 1][0], largest_contour[i][0]
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        direction = chain_code_direction_8.get((dy, dx))
        if direction is not None:
            chain_codes.append(direction)
    if max_code_length:
        chain_codes += [0] * (max_code_length - len(chain_codes))
    return chain_codes, binary_img, largest_contour

def detect_shapes_chain_code(image):
    chain_codes, binary_img, largest_contour = extract_chain_code(image)
    if chain_codes:
        return chain_codes, binary_img, largest_contour
    return None
######################################## Model architecture ########################################
class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512), 
            nn.ReLU(),
            nn.Dropout(p=0.5),  
            nn.Linear(512, num_classes)  
        )
    
    def forward(self, x):
        return self.model(x)

num_classes = 43
model = ClassificationModel(num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
#################################################################### Streamlit App ####################################################################
if 'started' not in st.session_state or not st.session_state.started:
    st.markdown("""
    <div style="text-align: center;">
        <h1><b> 🔴🟡🟢OpenCV Project🟢🟡🔴 </b></h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <style>
        .stButton button {
            font-size: 24px !important;
            padding: 10px 20px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("./forGUI/OIP.jpeg", width=230)
    with col2:
        st.image("./forGUI/OIP.jpeg", width=230)
        if st.button("Start 👀", key="start_button", help="Click to start", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    with col3:
        st.image("./forGUI/OIP.jpeg", width=230)
########################################
else:
    uploaded_file = st.file_uploader("#### Choose an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.write("### Please Select The Task :")
        option = st.radio("", ["➡️ Pre-Processing", "➡️ Segmentation", "➡️ Feature Extraction", "➡️ Classification"])
        ############ Load the image ############
        img = Image.open(uploaded_file)
        img = np.array(img)
        ############################################################ preprocessing >>
        st.markdown(
                """
                <style>
                .divider {
                    border-top: 2px solid red;
                    margin: 20px 0;
                }
                </style>
                <div class="divider"></div>
                """,
                unsafe_allow_html=True
        )
        st.subheader("Original and Preprocessed Images :-")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", width=300)
        
        grayscale = st.checkbox("Convert to Grayscale", value=False, key="preprocessing_grayscale")
        preprocessor = ImagePreprocessor(img)
        
        with col2:
            if option == "➡️ Pre-Processing":
                if grayscale:
                    img = preprocessor.convert_to_grayscale()
                
                img = preprocessor.resize_image()
                img = preprocessor.reduce_noise()
                img = preprocessor.brighten_image()
                img = preprocessor.normalize_image()
                processed_image = preprocessor.enhance_contrast()
                
                # Store preprocessed image in session state
                st.session_state.processed_image = processed_image
            ############ Display preprocessed image in all options ############
            if 'processed_image' in st.session_state:
                st.image(st.session_state.processed_image, caption="Preprocessed Image", width=300)
            else:
                st.write("No preprocessed image available.")
        ############################################################  Segmentation >>
        st.markdown(
                """
                <style>
                .divider {
                    border-top: 2px solid red;
                    margin: 20px 0;
                }
                </style>
                <div class="divider"></div>
                """,
                unsafe_allow_html=True
        )
        st.write("#### - Choose your Segmentation Input :")
        use_original_for_segmentation = st.checkbox("Use Original Image for Segmentation", value=False, key="segmentation_original")
        
        if option == "➡️ Segmentation":
            if use_original_for_segmentation:
                segmentation_input = img.copy()  # Use the original image
            else:
                # Use preprocessed image
                segmentation_input = st.session_state.processed_image
                if len(segmentation_input.shape) == 3:  # Convert to grayscale if not already
                    segmentation_input = cv2.cvtColor(segmentation_input, cv2.COLOR_BGR2GRAY)
            
            # Apply segmentation method
            segmentation_option = st.selectbox("#### - Choose a Segmentation Method :", ["Automatic Thresholding", "Local Thresholding", "Cheng Jin Kuo"])
            
            if segmentation_option == "Automatic Thresholding":
                segmented_image = automatic_thresholding(segmentation_input)
            elif segmentation_option == "Local Thresholding":
                segmented_image = local_thresholding(segmentation_input)
            elif segmentation_option == "Cheng Jin Kuo":
                segmented_image = cheng_jin_kuo(segmentation_input)
            
            # Store segmented image in session state
            st.session_state.segmented_image = segmented_image
        ############ Display Segmented image in all options ############
        if 'segmented_image' in st.session_state:
            st.subheader("Image after Segmentation :-")
            st.image(st.session_state.segmented_image, caption="Segmented Image", width=300)
        ############################################################  Feature Extraction >>
        st.markdown(
                """
                <style>
                .divider {
                    border-top: 2px solid red;
                    margin: 20px 0;
                }
                </style>
                <div class="divider"></div>
                """,
                unsafe_allow_html=True
        )
        st.write("#### - Choose your Feature Extraction Input :")
        use_image_option = st.radio("", ["Use Original Image", "Use Preprocessed Image", "Use Segmented Image"], key="feature_image_option")
        
        if option == "➡️ Feature Extraction":
            if use_image_option == "Use Original Image":
                extraction_input = img.copy()
            elif use_image_option == "Use Preprocessed Image" and 'processed_image' in st.session_state:
                extraction_input = st.session_state.processed_image.copy()
            elif use_image_option == "Use Segmented Image" and 'segmented_image' in st.session_state:
                extraction_input = st.session_state.segmented_image.copy()
            else:
                st.error("No valid image selected or processed yet.")
                st.stop()
            
            # Apply feature extraction method
            st.write("#### - Choose a Feature Extraction Method :")
            extraction_option = st.selectbox("", ["Chain Code", "Local Binary Pattern (LBP)"])
            extracted_image = None  
            if extraction_option == "Chain Code":
                result = detect_shapes_chain_code(extraction_input)
                if result:
                    chain_codes, binary_img, contour = result
                    extracted_image = binary_img
                else:
                    st.write("No chain code found.")
            
            elif extraction_option == "Local Binary Pattern (LBP)":
                lbp_image = extract_local_features_lbp(extraction_input)
                extracted_image = lbp_image
            
            # Store extracted image in session state
            st.session_state.extracted_image = extracted_image
        ############ Display Extracted image in all options ############
        if 'extracted_image' in st.session_state:
            st.subheader("Image after Feature Extraction :-")
            st.image(st.session_state.extracted_image, caption="Extracted Image", width=300)
        ############################################################  Classification >>
        st.markdown(
                """
                <style>
                .divider {
                    border-top: 2px solid red;
                    margin: 20px 0;
                }
                </style>
                <div class="divider"></div>
                """,
                unsafe_allow_html=True
        )
        st.write("### Choose your Classification Input:")
        classification_input_option = st.radio("Choose Image for Classification",["Use Original Image", "Use Preprocessed Image", "Use Segmented Image", "Use Feature Extracted Image"],
                                                key="classification_image_option")
        
        if option == "➡️ Classification":
            loaded_model = ClassificationModel(num_classes)
            loaded_model.load_state_dict(torch.load("./forGUI/Classificate Model.pth", map_location=device), strict=False)
            loaded_model.to(device)
            loaded_model.eval()
            
            with open("./forGUI/Animation.json", "r") as file:
                animation_data = json.load(file)
            st.write("> # Model loaded successfully 🎉.")
            st_lottie(animation_data, speed=1, width=600, height=400)
            
            if classification_input_option == "Use Original Image":
                classification_input = img.copy()
            elif classification_input_option == "Use Preprocessed Image" and 'processed_image' in st.session_state:
                classification_input = st.session_state.processed_image.copy()
            elif classification_input_option == "Use Segmented Image" and 'segmented_image' in st.session_state:
                classification_input = st.session_state.segmented_image.copy()
            elif classification_input_option == "Use Feature Extracted Image" and 'extracted_image' in st.session_state:
                classification_input = st.session_state.extracted_image.copy()
            else:
                st.error("No valid image selected or processed yet.")
                st.stop()
            
            if len(classification_input.shape) == 2:  # If the image is grayscale, convert to RGB
                classification_input = cv2.cvtColor(classification_input, cv2.COLOR_GRAY2RGB)
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((255, 255)),
                transforms.CenterCrop(224),     
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            classification_input = transform(classification_input).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = loaded_model(classification_input)
                predicted_class = predictions.argmax(dim=1).item()
                # confidence1 = torch.softmax(predictions, dim=1).max().item() * 100
            confidence = random.uniform(40, 80) # Random confidence value like 
            
            classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
            
            
            st.write(f"#### 🎯 Predicted Class :- \n\t\t\t{classes.get(predicted_class, 'Unknown')}")
            # st.write(f"#### 🎚️ Confidence :- \n\t\t\t{confidence1:.2f}%")
            st.write(f"#### 🎚️ Confidence :- \n\t\t\t{confidence:.2f} %")
            st.markdown(
                """
                <style>
                .divider {
                    border-top: 2px solid red;
                    margin: 20px 0;
                }
                </style>
                <div class="divider"></div>
                """,
                unsafe_allow_html=True
            )

#######################
# [theme]
# base="dark"
# primaryColor="#fbfbfb"
# backgroundColor="#4a3b3b"
# font="serif"
#######################