""" Service to make preprocessing accessible via http requests """
import os
from typing import List, Literal

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import uvicorn
import pandas as pd

from preprocessing.clean_features import KaggleFeatureCleaner
from preprocessing.transform_features import KaggleFeatureTransformer
from utils.data_models import (
    RequestInputInference,
    PreprocessedRequestInference,
    ExecutionMode,
)


app = FastAPI()


@app.post("/preprocess", response_model=List[PreprocessedRequestInference])
def preprocess_data(
    user_request: List[RequestInputInference], mode: Literal["inference"] = "inference"
) -> List[PreprocessedRequestInference]:
    """
    Preprocesses the posted input data.
    Args:
        user_request (List[``utils.data_models.PreprocessedRequestInference``]): List of request inputs to be processed.
        mode (str): Mode to execute the preprocessing. Allowable: 'inference'.

    Returns:
        List of preprocessed inputs.
    """
    execution_mode = ExecutionMode(mode)
    input_data = pd.DataFrame(jsonable_encoder(user_request))
    cleaner = KaggleFeatureCleaner(data=input_data, mode=execution_mode)
    cleaned_data = cleaner.execute()
    transformer = KaggleFeatureTransformer(
        data=cleaned_data,
        mode=execution_mode,
        labels_path=os.getenv("LABELS_PATH", None),
    )
    transformed_data = transformer.execute()
    # if execution_mode is ExecutionMode.TRAIN:
    #     target_cleaner = KaggleTargetCleaner(data=input_data)
    #     cleaned_targets = target_cleaner.execute()
    #     target_transformer = KaggleTargetTransformer(data=cleaned_targets)
    #     transformed_targets = target_transformer.execute()
    #     transformed_data = KaggleTrainDataLoader.match(features=transformed_data, targets=transformed_targets)
    return_data = transformed_data.to_dict(orient="records")
    return return_data


if __name__ == "__main__":
    uvicorn.run("preprocessing_service:app", log_level="info", port=8000)
