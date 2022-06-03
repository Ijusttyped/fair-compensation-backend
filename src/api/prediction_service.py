""" Service to make model predictions accessible via http requests """
import os
from typing import List
import logging

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import uvicorn
import pandas as pd

from modeling.sklearn_models import SKLearnModel
from utils.data_models import PreprocessedRequestInference, RequestOutput


app = FastAPI()
logger = logging.getLogger(os.getenv("LOGGER", "default"))


MODEL = SKLearnModel()
MODEL.load(os.getenv("MODEL_PATH", None))


@app.post("/predict", response_model=List[RequestOutput])
def predict(
    user_request: List[PreprocessedRequestInference],
) -> List[RequestOutput]:
    """
    Calculates predictions for a preprocessed input.
    Args:
        user_request (``utils.data_models.PreprocessedRequestInference``): Preprocessed data to calculate
            predictions for.

    Returns:
        List of calculated predictions.
    """
    logger.debug("Got request to prediction service: \n %s", user_request)
    input_data = pd.DataFrame(jsonable_encoder(user_request))
    predictions = MODEL.predict(input_data)
    request_output = [RequestOutput(Salary_Yearly=pred) for pred in predictions]
    return request_output


if __name__ == "__main__":
    uvicorn.run("prediction_service:app", log_level="info", port=8001)
