import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import bcrypt
import base64
from pathlib import Path
from datetime import datetime
import re
import pickle
import numpy as np
import time
user_data_file = "login_data.csv"
feedback_file = "feedback.csv"
# Ensure necessary files exist
def ensure_user_data():
    if not os.path.exists(user_data_file):
        df = pd.DataFrame(columns=['Username', 'Password'])
        df.to_csv(user_data_file, index=False)
def ensure_feedback_file():
    if not os.path.exists(feedback_file):
        pd.DataFrame(columns=["Name", "Age", "Gender", "Rating", "Feedback"]).to_csv(feedback_file, index=False)

ensure_user_data()
ensure_feedback_file()
# Load user data
def load_user_data():
    return pd.read_csv(user_data_file)
# Save new user data
def save_user_data(username, password):
    df = load_user_data()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = pd.DataFrame([[username, hashed_password]], columns=['Username', 'Password'])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(user_data_file, index=False)

# Check if the username exists
def username_exists(username):
    df = load_user_data()
    return not df[df['Username'] == username].empty

# Validate login
def validate_login(username, password):
    df = load_user_data()
    user_record = df[df['Username'] == username]
    if not user_record.empty:
        stored_hashed_password = user_record['Password'].values[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8'))
    return False

# Change password functionality
def change_password(username, new_password):
    df = load_user_data()
    if username_exists(username):
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        df.loc[df['Username'] == username, 'Password'] = hashed_password
        df.to_csv(user_data_file, index=False)
        return True
    return False
# Save feedback data to CSV
def save_feedback(name, age, gender, rating, feedback):
    rating_map = {
        1: "1 Star - Poor",
        2: "2 Stars - Fair",
        3: "3 Stars - Average",
        4: "4 Stars - Good",
        5: "5 Stars - Excellent"
    }
    formatted_rating = rating_map[rating]
    
    
    feedback_data = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Rating": [formatted_rating],
        "Feedback": [feedback]
    })

    if os.path.exists(feedback_file):
        existing_data = pd.read_csv(feedback_file)
        feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)

    feedback_data.to_csv(feedback_file, index=False)
# Streamlit app title
st.title("Fraud Detection Using Deep Learning Explainable AI (XAI) 💵")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""  # Initialize username as an empty string

# Display logout message if logged out
if 'logout_message' in st.session_state:
    st.success(st.session_state.logout_message)
    del st.session_state.logout_message  # Clear the message after displaying it
# Display login interface only if not logged in
if not st.session_state.logged_in:
    # Set the background image for the login interface
    def set_login_background(image_file):
        login_bg_img = f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(login_bg_img, unsafe_allow_html=True)

    # Load the background image for the login interface
    with open("digital-art.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background for the login interface
    set_login_background(encoded_image)

    with st.expander("Authentication", expanded=True):
        menu = option_menu(
                            menu_title=None,
                            options=['Login', 'Register', 'Forgot Password'],
                            icons=['box-arrow-right', 'person-plus', 'key'],
                            orientation='horizontal'
  
                          )

        if menu == 'Register':
            st.subheader('Register')
            username = st.text_input("Choose a Username", key="register_username")  # Unique key for username
            password = st.text_input("Choose a Password", type="password", key="register_password")  # Unique key for password
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")  # Unique key for confirm password

            def is_valid_password(password):
                # Check if password meets the requirements
                if len(password) < 8 or len(password) > 12:
                    st.error("Password must be between 8 to 12 characters length.")
                    return False
                if not any(char.isupper() for char in password):
                    st.error("Password must include at least one uppercase letter (A-Z).")
                    return False
                if not any(char.islower() for char in password):
                    st.error("Password must include at least one lowercase letter (a-z).")
                    return False
                if not any(char.isdigit() for char in password):
                    st.error("Password must include at least one digit (0-9).")
                    return False
                if not re.search(r'[!@#$%^&*]', password):
                    st.error("Password must include at least one special character (!@#$%^&*).")
                    return False
                return True

            if st.button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif username_exists(username):
                    st.error("Username already exists. Please choose a different one.")
                elif not is_valid_password(password):
                    # Password requirements not met, error messages will be displayed in is_valid_password
                    pass
                else:
                    save_user_data(username, password)
                    st.success("Registration successful! You can now log in.")

        elif menu == 'Forgot Password':
            st.subheader('Reset Password')
            username = st.text_input("Enter your Username")
            new_password = st.text_input("Enter your New Password", type='password')
            confirm_password = st.text_input("Confirm New Password", type='password')

            if st.button("Reset Password"):
                if username_exists(username):
                    if new_password == confirm_password:
                        if change_password(username, new_password):
                            st.success("Your password has been reset successfully.")
                        else:
                            st.error("Failed to reset password. Please try again.")
                    else:
                        st.error("Passwords do not match! Please try again.")
                else:
                    st.error("Username not found!")

        elif menu == 'Login':
            st.subheader('Login')
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if validate_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username  # Store the username in session state
                    st.session_state.show_success_message = True  # Flag to show success message
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
# Main project interface
if st.session_state.logged_in:
    st.markdown(f"## Welcome, {st.session_state.username}!", unsafe_allow_html=True)
    # Set the background image for the Streamlit interface
    def set_background(image_file):
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Load the background image for the interface
    with open("dark-abstract.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background
    set_background(encoded_image)
    
    # Display the "Login successful!" message if the user has just logged in
    if 'show_success_message' in st.session_state and st.session_state.show_success_message:
        st.success("Login successful!✅")
        time.sleep(3)  # Keep the message for 3 seconds
        st.session_state.show_success_message = False
        st.experimental_rerun() # Reset the flag

    # Loading the saved model
    loaded_model = pickle.load(open("final_model.sav", 'rb'))

    # Creating a function for Prediction
    @st.cache(persist=True)
    def predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType):
        input = np.array([[card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType]])
        prediction = loaded_model.predict_proba(input)
        pred = '{0:.{1}f}'.format(prediction[0][0], 2)
        return float(pred)

    def main():
        html_temp = """
            <div style="background-color:#000000 ;padding:10px">
            <h1 style="color:white;text-align:center;">Fraud Detection Using Deep Learning Explainable AI (XAI) Web App 💰 </h1>
            </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Displaying an image
        image = Image.open("home_banner.PNG")
        st.image(image, caption='Impacting the World of Finance and Banking with Artificial Intelligence (AI)')


    with st.sidebar:
        selected=option_menu(
            menu_title='Main Menu',
            options=['Detect Fraud', 'Feedback', 'About Us', 'Logout'],
            icons=['cash-coin', 'star-fill', 'person-circle', 'box-arrow-left'],
            menu_icon='cast',
            default_index=0
                            )
        
    if selected == 'Detect Fraud':
        #start logic here
        st.subheader("Financial Transaction Fraud Prediction System 🕵️")
        
        # Getting the input data from the user
        # Transaction Amount
        TransactionAmt = st.number_input("Choose the Transaction Amount in USD", 0, 20000, step=1)

        # Card 1 
        card1 = st.number_input("Choose the Payment Card 1 Amount (USD)", 0, 20000, step=1)

        # Card 2 
        card2 = st.number_input("Choose the Payment Card 2 Amount (USD)", 0, 20000, step=1)

        # Card Category
        card4 = st.radio("Choose the Payment Card Category", [1, 2, 3, 4])
        st.info("1 : Discover | 2 : Mastercard | 3 : American Express | 4 : Visa")

        # Card Type
        card6 = st.radio("Choose the Payment Card Type", [1, 2])
        st.info("1 : Credit | 2 : Debit")

        # Billing Zip Code
        addr1 = st.slider("Choose the Payment Billing Zip Code", 0, 500, step=1)

        # Billing Country Code
        addr2 = st.slider("Choose the Payment Billing Country Code", 0, 100, step=1)

        # Purchaser Email Domain
        P_emaildomain = st.selectbox("Choose the Purchaser Email Domain", [0, 1, 2, 3, 4])
        st.info("0 : Gmail (Google) | 1 : Outlook (Microsoft) | 2 : Mail.com | 3 : Others | 4 : Yahoo")

        # Product Code
        ProductCD = st.selectbox("Choose the Product Code", [0, 1, 2, 3, 4])
        st.info("0 : C | 1 : H | 2 : R | 3 : S | 4 : W")

        # Device Type
        DeviceType = st.radio("Choose the Payment Device Type", [1, 2])
        st.info("1 : Mobile | 2 : Desktop")

        safe_html = """ 
            <img src="https://media.giphy.com/media/g9582DNuQppxC/giphy.gif" alt="confirmed" style="width:698px;height:350px;"> 
            """
        
        danger_html = """  
            <img src="https://media.giphy.com/media/8ymvg6pl1Lzy0/giphy.gif" alt="cancel" style="width:698px;height:350px;">
            """

        # Creating a button for Prediction   
        if st.button("Click Here To Predict"):
            output = predict_fraud(card1, card2, card4, card6, addr1, addr2,
                                    TransactionAmt, P_emaildomain,
                                    ProductCD, DeviceType)
            final_output = output * 100
            st.subheader('Probability Score of Financial Transaction is {}% '.format(final_output))

            if final_output > 75.0:
                st.markdown(danger_html, unsafe_allow_html=True)
                st.error("**OMG! Financial Transaction is Fraud**")
            else:
                st.balloons()
                time.sleep(5)
                st.balloons()
                time.sleep(5)
                st.balloons()
                st.markdown(safe_html, unsafe_allow_html=True)
                st.success("**Hurray! Transaction is Legitimate**")

    if __name__ == '__main__':
        main()
        
    if selected == 'Feedback':
        st.title("Feedback")
        st.subheader("Give Us Your Feedback")

        user_name = st.text_input("Enter Your Name:")
        user_age = st.number_input("Enter Your Age:", min_value=1, max_value=120, step=1, format="%d")

        gender_options = ["Male", "Female"]
        selected_gender = st.selectbox("Select Your Gender:", gender_options)

        #feedback_rating = st.radio("Rate your experience (1-5 stars):", range(1, 6))
        # Rating selection
        feedback_rating = st.radio(
            "Rate Your Experience (1-5 stars):",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 Star - Poor",
                2: "2 Stars - Fair",
                3: "3 Stars - Average",
                4: "4 Stars - Good",
                5: "5 Stars - Excellent"
            }[x]
        )

        feedback_text = st.text_area("Share Your Suggestions: (if any)")

        if st.button("Submit Feedback"):
            if user_name and user_age and selected_gender:
                
                save_feedback(user_name, user_age, selected_gender, feedback_rating, feedback_text)
                st.success("Thank you for your feedback!")
            else:
                st.error("Please fill in all fields before submitting.")
    if selected == 'About Us':
        # Add Font Awesome CDN link to your Streamlit app
        st.markdown(
            """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            """,
            unsafe_allow_html=True
        )
        st.markdown("## <i class='fas fa-info-circle'></i> About Us", unsafe_allow_html=True)
        st.write("We are a dedicated team committed to providing the best service.")

        
        st.markdown("<h3><i class='fas fa-bullseye'></i> Our Mission:</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. <i class='fas fa-robot'></i> **Develop an Explainable AI (XAI)-driven Interface** for financial transaction fraud detection that enhances transparency and trust in AI predictions, enabling users to understand how decisions are made.
        2. <i class='fas fa-chart-line'></i> **Implement Robust Machine Learning Models** to accurately classify financial transactions as fraudulent or legitimate using advanced algorithms, ensuring high performance and reliability.
        3. <i class='fas fa-check-circle'></i> **Evaluate Model Performance** through comprehensive metrics such as Accuracy, AUC-ROC Score, Precision, Recall, and F1 Score, ensuring that our models meet industry standards.
        4. <i class='fas fa-lightbulb'></i> **Integrate Explainability Methods** like SHAP and LIME to provide insights into model predictions, making the decision-making process more understandable for business stakeholders.
        5. <i class='fas fa-user-friends'></i> **Collect and Analyze User Feedback** to continuously improve the application’s features and user experience based on real-world interactions.
        6. <i class='fas fa-desktop'></i> **Develop a Proof of Concept (POC) Web Application** that allows business decision-makers to make real-time predictions on fraud detection, generating actionable insights and value.
        7. <i class='fas fa-info-circle'></i> **Provide an 'About Us' Section** to inform users about the team behind the project and our vision for leveraging AI in financial services.
        8. <i class='fas fa-sign-out-alt'></i> **Offer a 'Logout' Feature** to ensure secure and personalized user sessions, protecting sensitive financial information.
        """, unsafe_allow_html=True)

        # Team Section
        st.markdown("<h3><i class='fas fa-users'></i> The Team:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - <i class='fas fa-user'></i> **BOPPANA ROHITH (99220041454), Kalasalingam Academy of Research and Education, CSE -- Team Lead and Developer** - Specializes in Deep Learning and Artificial intelligence algorithms.
        - <i class='fas fa-user'></i> **ANIMMA SRINIVASINE P (99220041437), Kalasalingam Academy of Research and Education, CSE -- Researcher** - Provides domain expertise on Fraud Detection.
        - <i class='fas fa-user'></i> **BACHULA YASWANTH BABU (99220041445), Kalasalingam Academy of Research and Education, CSE -- Web Developer** - Responsible for designing and implementing the project's web application.
        - <i class='fas fa-user'></i> **ANISETTY.SAI PRAJWIN (99220041438), Kalasalingam Academy of Research and Education, CSE -- Data Scientist** - Analyzes data and develops insights to improve the features of Project.
        """, unsafe_allow_html=True)

        # Project Mentor
        st.markdown("<h3><i class='fas fa-chalkboard-teacher'></i> Faculty Instructor:</h3>", unsafe_allow_html=True)
        st.write("**Mr. N. R. SATHIS KUMAR, Kalasalingam Academy of Research and Education, CSE**")

        ## Project Evaluator
        #st.markdown("<h3><i class='fas fa-pen'></i> Project Evaluator:</h3>", unsafe_allow_html=True)
        #st.write("**XYZ**")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3><i class='fas fa-star'></i> We are dedicated to leveraging Explainable AI (XAI) to enhance transparency and trust in financial transaction fraud detection. This project is part of our commitment to advancing AI-driven solutions, offering robust deep learning models, comprehensive performance evaluation, and intuitive explainability methods to empower business decision-makers and prevent fraud in the financial services industry !!</h3>", unsafe_allow_html=True)
    # Logout button
    if selected == 'Logout':
        st.session_state.logged_in = False
        st.session_state.logout_message = "You have successfully logged out! Please log in again to continue exploring our Fraud Detection System."
        st.experimental_rerun() # Refresh the page
