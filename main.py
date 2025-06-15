import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index
# Sidebar
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sidebar .css-1d391kg {  /* Title text */
        font-size: 30px;
        color: #4CAF50;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .sidebar .css-1cpxqw2 {
        font-size: 18px;
        color: #37474F;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## 📊 Dashboard")

app_mode = st.sidebar.selectbox(
    "🌐 Select a Page",
    ["🏠 Home", "ℹ️ About", "🦠 Disease Recognition"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Choose a page to begin exploring!*")


#Home page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿</h1>", unsafe_allow_html=True)
    
    # Display image
    image_path = "leaf image.jpg"
    st.image(image_path, caption="Empowering Farmers with AI-Based Plant Disease Detection", use_container_width=True)

    # Intro section
    st.markdown("""
    <div style='font-size: 18px; line-height: 1.6;'>
        Welcome to the <strong>Plant Disease Recognition System</strong>! 🌿🔍<br>
        Our mission is to help farmers and researchers <strong>identify plant diseases quickly and accurately</strong>. 
        Simply upload a leaf image, and our smart system will analyze it for disease symptoms — making farming smarter and healthier! 🌾
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # How It Works
    st.markdown("## 🛠️ How It Works")
    st.markdown("""
    1. 📸 **Upload Image:** Navigate to the **🦠 Disease Recognition** page and upload an image of the affected plant leaf.
    2. 🧠 **Analysis:** Our intelligent model will examine the image using advanced deep learning algorithms.
    3. ✅ **Results:** Receive instant predictions and disease details, along with suggested next steps.
    """)

    st.markdown("---")

    # Why Choose Us
    st.markdown("## 🌟 Why Choose Us?")
    st.markdown("""
    - 🎯 <strong>Accuracy:</strong> Powered by cutting-edge machine learning models trained on thousands of images.
    - 🎨 <strong>User-Friendly:</strong> Minimalist and intuitive interface — no tech knowledge needed.
    - ⚡ <strong>Fast & Efficient:</strong> Get results in seconds, saving crucial response time in the field.
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Call to Action
    st.markdown("## 🚀 Get Started")
    st.markdown("""
    Head over to the **🦠 Disease Recognition** page in the sidebar to begin! Upload a plant leaf image and let our system do the work. 👨‍🌾🤖
    """)

    st.markdown("---")

   #About 

elif (app_mode == "ℹ️ About"):
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 About the Dataset</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 17px; line-height: 1.6;'>
        Welcome to the enhanced crop leaf image dataset! This resource has been meticulously curated using <b>offline data augmentation</b> techniques based on the <a href='https://github.com/spMohanty/PlantVillage-Dataset' target='_blank'>original dataset</a>.
        It contains over <b>87,000 high-quality RGB images</b> of healthy and diseased crop leaves 🌿, grouped into <b>38 distinct classes</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📊 Dataset Structure")
    st.markdown("""
    - **🧪 Train Set**: `70,295` images — used for model training  
    - **🧪 Validation Set**: `17,572` images — for evaluating model performance  
    - **🧪 Test Set**: `33` images — used for final predictions
    """)

    st.markdown("---")

    st.markdown("## 📂 Content Breakdown")
    st.markdown("""
    > 📁 **Train Set**  
    &nbsp;&nbsp;&nbsp;&nbsp;→ 70,295 images across 38 classes

    > 📁 **Validation Set**  
    &nbsp;&nbsp;&nbsp;&nbsp;→ 17,572 images for performance tuning

    > 📁 **Test Set**  
    &nbsp;&nbsp;&nbsp;&nbsp;→ 33 unseen images for final testing
    """)

    st.markdown("---")

    st.markdown("## 🎯 Key Highlights")
    st.markdown("""
    - 🧬 <b>Augmented data</b> ensures better generalization and diversity  
    - 🌾 Includes a wide range of <b>crop types and disease conditions</b>  
    - 🗂️ Structured directory format — ideal for deep learning workflows  
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📌 Why This Dataset?")
    st.markdown("""
    This dataset is a valuable asset for:
    - 🤖 **AI/ML model training**
    - 🌱 **Precision agriculture tools**
    - 🧪 **Computer vision research**
    
    It empowers developers and researchers to build robust disease detection systems, improve yield forecasts, and contribute to sustainable farming practices.

    > <i>“Detect early, act fast — save crops, save harvests.”</i>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 🔗 Original Dataset Link")
    st.markdown("[🔍 View Original Dataset on GitHub](https://github.com/spMohanty/PlantVillage-Dataset)")

    

# Prediction page
elif(app_mode == "🦠 Disease Recognition"):
    st.markdown("<h1 style='color: #2E8B57;'>🧪 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("Upload a leaf image to identify possible diseases using our AI model. 🌿")

    # File uploader without restricting file type
    test_image = st.file_uploader("📂 Choose an image:")

    # Show image button
    if st.button("📸 Show Image"):
        if test_image is not None:
            st.image(test_image, caption="Uploaded Image", width=300)
        else:
            st.warning("⚠️ Please upload an image before clicking 'Show Image'.")

    # Predict button
    if st.button("🤖 Predict"):
        if test_image is not None:
            st.markdown("### 🔍 Making the Prediction...")
            # Show the spinner with a custom message
            with st.spinner("🔄 Processing... Please wait while we analyze the image."):
                result_index = model_prediction(test_image)

            # Define class
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            st.image(test_image, caption="Uploaded Image", width=300)
            st.success(f"🌟 Model predicts: **{class_name[result_index]}**")
        else:
            st.error("🚫 No image uploaded! Please upload an image first.")
