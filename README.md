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



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - [@khalidasdsju](https://github.com/khalidasdsju)

Project Link: [https://github.com/khalidasdsju/mlfow-test](https://github.com/khalidasdsju/mlfow-test)

## MLflow on AWS

### MLflow on AWS Setup

1. **Login to AWS Console**:
   - Create an IAM user with `AdministratorAccess`.
   - Export the credentials in your AWS CLI by running:
	 ```bash
	 aws configure
	 ```

2. **Create an S3 Bucket**:
   - Use the AWS Management Console or CLI to create a bucket for storing MLflow artifacts.

3. **Set Up an EC2 Machine**:
   - Launch an Ubuntu EC2 instance.
   - Add a security group rule to allow inbound traffic on port `5000`.

4. **Install Required Tools on EC2**:
   Run the following commands on the EC2 instance:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   sudo apt install pipenv
   sudo apt install virtualenv
   ```

5. **Prepare the MLflow Directory**:
   ```bash
   mkdir mlflow
   cd mlflow
   pipenv install mlflow awscli boto3
   pipenv shell
   ```

6. **Set AWS Credentials**:
   ```bash
   aws configure
   ```

7. **Start the MLflow Server**:
   ```bash
   mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-test-23
   ```

8. **Access the MLflow Server**:
   - Open the EC2 instance's Public IPv4 DNS on port `5000`.

9. **Set the Tracking URI Locally**:
   ```bash
   export MLFLOW_TRACKING_URI=https://ec2-3-95-237-202.compute-1.amazonaws.com:5000/
   ```
   Replace `<EC2-Public-DNS>` with the actual Public IPv4 DNS of your EC2 instance.