# frontend/app.py
import streamlit as st
import requests

# FastAPI backend endpoint
API_URL = "https://o1a-fastapi.onrender.com/full-assessment/"

st.set_page_config(page_title="O-1A Visa Assessment", layout="centered")

st.title("O-1A Visa Assessment Tool")
st.write("Upload your resume PDF to get an O-1A assessment rating and personalized recommendations.")

# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Uploading and processing..."):
        try:
            # Send the file to the FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(API_URL, files=files)

            # Check response status
            if response.status_code == 200:
                data = response.json()

                # Extract the final assessment
                final_assessment = data.get("assessment_result", {}).get("final_assessment", {})

                rating = final_assessment.get("rating", "No rating found")
                recommendations = final_assessment.get("recommendations", "No recommendations found")
                # Determine the color based on the rating
                if rating == "LOW":
                    color = "red"
                elif rating == "MODERATE":
                    color = "orange"
                elif rating == "HIGH":
                    color = "green"
                else:
                    color = "black"
                # Display results
                # Display the rating and recommendations
                st.subheader("Assessment Result")
                st.markdown(f"<p style='color:{color}; font-size:24px;'>Rating: {rating}</p>", unsafe_allow_html=True)
                st.write(f"Recommendations: {recommendations}")
            

                # Optional: Expand to show full JSON
                with st.expander("Show Full JSON Response"):
                    st.json(data)

            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
