import streamlit as st
import requests

# FastAPI backend endpoint
API_URL = "http://localhost:8000/full-assessment/"

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

                # Display the rating and recommendations
                st.subheader("Assessment Result")
                st.markdown(f"<p style='color:{color}; font-size:24px;'>Rating: {rating}</p>", unsafe_allow_html=True)
                

                # --- NEW SECTION: Show all 8 Criteria with evidence and justification ---
                st.subheader("O-1A Criteria Breakdown")

                # Extract child assessments
                child_assessments = data.get("assessment_result", {}).get("child_assessments", {})

                if not child_assessments:
                    st.warning("No child assessments found.")
                else:
                    for criterion_key, criterion_data in child_assessments.items():
                        # Extract criterion details
                        criterion_name = criterion_data.get("criterion_mapping", {}).get("criterion", "Unknown Criterion")
                        evidence_items = criterion_data.get("assessment", {}).get("evidence_items", [])
                        justification = criterion_data.get("assessment", {}).get("justification", "No justification provided.")

                        # Display Criterion Name
                        st.markdown(f"### {criterion_name}")

                        # Display Evidence Items
                        if evidence_items:
                            for idx, item in enumerate(evidence_items, start=1):
                                description = item.get("description", "No description available.")
                                source = item.get("source", "No source available.")
                                strength = item.get("strength", "No strength rating.")

                                st.markdown(f"**Evidence {idx}:**")
                                st.write(f"- **Description:** {description}")
                                st.write(f"- **Source:** {source}")
                                st.write(f"- **Strength:** {strength}")
                        else:
                            st.info("No evidence items found.")

                        # Display Justification
                        st.markdown("**Justification:**")
                        st.write(justification)

                        # Spacer for readability
                        st.markdown("---")
                        st.markdown("---")

                st.subheader("Recommendations")
                st.write(recommendations)
                

                # Optional: Expand to show full JSON
                with st.expander("Show Full JSON Response"):
                    st.json(data)

            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
