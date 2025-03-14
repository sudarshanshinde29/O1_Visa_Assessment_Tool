# O-1A Visa Assessment Tool

This repository contains a tool for assessing O-1A visa eligibility based on resume data. The tool consists of a FastAPI backend for processing resumes and a Streamlit frontend for user interaction.

## Table of Contents

- [O-1A Visa Assessment Tool](#o-1a-visa-assessment-tool)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up the FastAPI Backend](#2-set-up-the-fastapi-backend)
    - [3. Set Up the Streamlit Frontend](#3-set-up-the-streamlit-frontend)
  - [Usage](#usage)
    - [Running the FastAPI Backend](#running-the-fastapi-backend)
    - [Running the Streamlit Frontend](#running-the-streamlit-frontend)

## Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) (optional but recommended)

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/sudarshanshinde29/O1_Visa_Assessment_Tool.git
cd O1_Visa_Assessment_Tool
```

### 2. Set Up the FastAPI Backend

Create a virtual environment and activate it:
```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies:
```sh
pip install -r requirements.txt
```

Set up environment variables:
Create a .env file in the root directory and add the following variables:
```sh
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
OPENAI_API_KEY=your_openai_api_key
```
Start the FastAPI server:
```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Set Up the Streamlit Frontend
Navigate to the frontend directory:
```sh
cd frontend
```

Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Running the FastAPI Backend

To run the FastAPI backend, use the following command:

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at http://localhost:8000.

### Running the Streamlit Frontend

To run the Streamlit frontend, use the following command:

```sh
streamlit run streamlit.app.py
```

The frontend will be available at http://localhost:8501.