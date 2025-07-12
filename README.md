# IRIS-CLASSIFICATION-WITH-RF-MODEL

In this project, we develop a machine learning model to classify Iris flower species using the Random Forest (RF) algorithm. The Iris dataset, consisting of 150 samples with four features (sepal length, sepal width, petal length, petal width), serves as the basis for training and testing our model. The dataset is preprocessed to handle any missing values and encode categorical labels. We split the data into training and testing sets to evaluate model performance. The Random Forest classifier, known for its robustness and accuracy, is then trained and its hyperparameters are tuned for optimal performance. Finally, we assess the model's accuracy, precision, recall, and F1-score to ensure it accurately predicts the species of Iris flowers. The project demonstrates the effectiveness of Random Forests in handling classification tasks with structured data.

## Getting Started

### Prerequisites

*   Python 3.7+
*   pip
*   virtualenv (recommended)

### Setup and Running the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sriharish120/IRIS-CLASSIFICATION-WITH-RF-MODEL.git
    cd IRIS-CLASSIFICATION-WITH-RF-MODEL
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .env
    source .env/bin/activate  # On Windows, use `.env\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

### Running with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t iris-classification-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 iris-classification-app
    ```
    The application will be available at `http://localhost:8501`.
