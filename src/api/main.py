""" Service to orchestrate the microservices """
from typing import List

from fastapi import FastAPI
import uvicorn

from api.preprocessing_service import preprocess_data
from api.prediction_service import predict
from utils.data_models import RequestInputInference, RequestOutput


app = FastAPI()


@app.post("/get_salary", response_model=List[RequestOutput])
def preprocess_and_predict(
    user_request: List[RequestInputInference],
) -> List[RequestOutput]:
    """
    Preprocessed the input and calculates predictions for the preprocessed data.
    Args:
        user_request (``utils.data_models.RequestInputInference``): Raw input to preprocess and calculate
            predictions for.

    Returns:
        List of calculated predictions.
    """
    preprocessed_data = preprocess_data(user_request=user_request)
    predictions = predict(user_request=preprocessed_data)
    return predictions


if __name__ == "__main__":
    uvicorn.run("main:app", log_level="info")
