# Car Accident Detection Web Project

This project is a web application that detects car accidents using a machine learning model. The application is built using Python and utilizes Flask for the web framework.

## Project Structure

```
car-accident-detection-web
├── src
│   ├── app.py                # Main application file
│   ├── model
│   │   └── best.pt           # Trained machine learning model
│   └── templates
│       └── index.html        # HTML structure for the web page
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd car-accident-detection-web
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```
   python src/app.py
   ```

2. **Access the web application**:
   Open your web browser and go to `http://127.0.0.1:5000` to view the application.

## Model Information

The model used for detecting car accidents is located in `src/model/best.pt`. It is a PyTorch model that has been trained to identify car accidents based on input data.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. 

## License

This project is licensed under the MIT License.