
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache()
def load_model():
    model = tf.keras.models.load_model("./model/mobilenetV2/mobilenetv2.h5", compile=False)
    return model

# Function to make predictions on the uploaded image
def predict_image_class(image, model):
    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)[0]
    return predictions

# Function to check if the image contains chicken feces
def is_chicken_feces_image(predictions, threshold=0.5):
    # Define the classes
    classes = ['Coccidiosis', 'Healthy', 'NCD', 'Salmonella']

    # Check if any of the predictions indicate a disease
    for i, prediction in enumerate(predictions):
        if classes[i] != 'Healthy' and prediction >= threshold:
            return True, classes[i]

    return False, None

# Define data about diseases
disease_data = {
    'Coccidiosis': {
        'medication': [
            'Amprolium',
            'Sulphonamides',
            'Toltrazuril'
        ],
        'treatment': 'Treatment for coccidiosis includes medication prescribed by a veterinarian or anticoccidial medication available at feed stores.',
        'prevention': 'Prevent coccidiosis by maintaining good sanitation in the poultry house, using approved anticoccidial medications, and vaccinating birds.',
        'causes': 'Coccidiosis is caused by the ingestion of feces from infected birds, which contain the parasite that causes the disease.'
    },
    'Salmonella': {
        'medication': [
            'Neomycin',
            'Neomycin plus oxytetracycline',
            'Sulfadizine plus trimethoprim'
        ],
        'treatment': 'Treatment for salmonella may include medication administered by a veterinarian and supportive care.',
        'prevention': 'Prevent salmonella by implementing hygiene measures, limiting contact with infected animals, and maintaining a clean environment.',
        'causes': 'Salmonella infections in chickens can occur through contaminated feed, infected animals, or contact with a contaminated environment.'
    },
    'NCD': {
        'medication': [
            'Amprolium',
            'Sulphonamides',
            'Toltrazuril'
        ],
        'treatment': 'Treatment for Newcastle disease includes isolating infected birds, consulting a veterinarian, providing supportive care, and maintaining up-to-date vaccinations.',
        'prevention': 'Prevent Newcastle disease through vaccination, biosecurity measures, and monitoring for symptoms.',
        'causes': 'Newcastle disease is caused by contact with infected birds or contamination of the environment with the virus.'
    },
    'Healthy': {
        'medication': [],
        'treatment': 'No specific medication or treatment required.',
        'prevention': 'Maintain good hygiene, provide proper nutrition, and monitor the health of the chickens regularly.',
        'causes': 'Healthy chickens do not exhibit symptoms of diseases.'
    }
}

# Main function
def main():
    # Load the model
    model = load_model()

    # Title
    st.title('Fecal Chicken Disease Diagnostics🐣🐓💩')

    # File uploader
    file = st.file_uploader("Upload an image of chicken feces")

    if file is not None:
        # Display the uploaded image
        uploaded_image = Image.open(file)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Make predictions on the image
        predictions = predict_image_class(uploaded_image, model)

        # Check if the image contains chicken feces
        is_feces_image, predicted_class = is_chicken_feces_image(predictions)

        # Display the prediction result
        if is_feces_image:
            st.success(f'The image is classified as {predicted_class}')

            # Display additional information about the disease
            st.subheader('Additional Information:')
            st.write('**Medication:**')
            for medication in disease_data[predicted_class]['medication']:
                st.write(f"- {medication}")
            st.write('**Treatment:**', disease_data[predicted_class]['treatment'])
            st.write('**Prevention:**', disease_data[predicted_class]['prevention'])
            st.write('**Causes:**', disease_data[predicted_class]['causes'])
        else:
            if predicted_class == 'Healthy':
                st.success('The uploaded image indicates healthy chicken feces.')
            else:
                st.error('Error: The uploaded image does not contain chicken feces.')
            st.write('Please upload an image of chicken feces.')

# Run the app below
if __name__ == '__main__':
    main()
