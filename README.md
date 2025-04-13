# Random Forest Regression with MLflow

This project demonstrates how to train a Random Forest Regression model and track experiments using MLflow with DagsHub integration.

## Project Structure

```
mlfow-test/
├── README.md
├── example.py
├── terminal_command.sh
├── mlruns/
└── mlflow/
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/khalidasdsju/mlfow-test.git
cd mlfow-test
```

2. Create and activate a virtual environment:
```bash
python -m venv mlflow
source mlflow/bin/activate  # On Windows: mlflow\Scripts\activate
```

3. Install dependencies:
```bash
pip install mlflow pandas numpy scikit-learn
```

4. Set up MLflow tracking URI and credentials:
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/khalidasdsju/mlfow-test.mlflow"
export MLFLOW_TRACKING_USERNAME="khalidasdsju"
export MLFLOW_TRACKING_PASSWORD="your-token"
```

## Usage

1. Run the training script:
```bash
python example.py
```

2. View experiments in MLflow UI:
- Visit: https://dagshub.com/khalidasdsju/mlfow-test.mlflow

## Features

- Synthetic regression data generation
- Random Forest Regression model training
- Automatic model versioning
- Experiment tracking with MLflow
- Integration with DagsHub
- Performance metrics tracking (MSE, RMSE)

## Model Parameters

The Random Forest Regressor is configured with the following parameters:
- n_estimators: 100
- criterion: squared_error
- max_features: sqrt
- And more...

## Metrics Tracked

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## MLflow Integration

The project uses MLflow to track:
- Model parameters
- Performance metrics
- Model versions
- Model artifacts
- Run information

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - [@khalidasdsju](https://github.com/khalidasdsju)

Project Link: [https://github.com/khalidasdsju/mlfow-test](https://github.com/khalidasdsju/mlfow-test)
