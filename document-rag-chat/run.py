# Streamlit App Launcher

# run.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY not found. Please set it in your .env file or environment variables."
    )

# Import and run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.bootstrap

    streamlit.web.bootstrap.run("app.py", "", [], [])