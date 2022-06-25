# Fair Compensation

A showcase for using machine learning in human resources.

<br>

![Code-Quality](https://github.com/Ijusttyped/fair-compensation-backend/actions/workflows/code-quality.yaml/badge.svg)
![Unit-Tests](https://github.com/Ijusttyped/fair-compensation-backend/actions/workflows/tests.yaml/badge.svg)
![Build](https://github.com/Ijusttyped/fair-compensation-backend/actions/workflows/build.yaml/badge.svg)
![Deployment](https://github.com/Ijusttyped/fair-compensation-backend/actions/workflows/deployment.yaml/badge.svg)

## Project Goal

The goal of the project is to showcase how machine learning can be used to make the salary structure fair.
This may involve, for example, identifying and eliminating potential gender pay disparities,
making pay increases or new hires fair.

A live demo of the model in action can be visited on [ijusttyped.github.io/fair-compensation-frontend](http://ijusttyped.github.io/fair-compensation-frontend).

## Project Setup

#### Project Structure

The project is structured as follows:

```
├── .dvc                    <- Metadata for DVC.
├── .github                 <- Github actions for CI/CD.
├── artefacts               <- Artefacts produced when executing the model training.
├── data
│   └── interim             <- Intermediate data that has been transformed.
├── src
│   ├── api                 <- Package to provide projects functionality as callable API.
│   ├── data_loading        <- Functionality to load data.
│   ├── modelling           <- Functionality to build and use ML models.
│   ├── preprocessing       <- Functionality to preprocess data for modelling.
│   └── utils               <- Common functions used throughout the project.
├── docker-compose.yaml     <- Docker compose for testing the API's locally.
├── Dockerfile              <- File to build the docker container for the API.
├── dvc.yaml                <- DVC pipeline to execute the model training.
├── LICENSE
├── README.md               <- The top-level README for developers using this project.
└── requirements.txt        <- The project dependencies.
```

#### Environment Setup

The recommended way of installing the project's dependencies is to use a virtual environment.
To create a virtual environment, run the venv module inside the repository:

`python3 -m venv venv`

Once you have created a virtual environment, you may activate it by running: `source ./venv/bin/activate`

To install the dependencies, run: `python3 -m pip install -r requirements.txt`

For code execution, make sure that the `src` directory is part of your `PYTHONPATH`: `export PYTHONPATH=$PWD:$PWD/src/`


## Reproducing the Results

#### Raw Data

The data used in the project can be downloaded on [Kaggle](https://www.kaggle.com/datasets/parulpandey/2020-it-salary-survey-for-eu-region).
The data has to be placed in the `data/raw` folder.

#### Data Versioning

We use [DVC](https://dvc.org) to version the raw data, interim results and artefacts.

#### Model Training Pipeline

We use the pipelining functionality of [DVC](https://dvc.org) to streamline the model training.
The pipeline is structured as follows:

```
                  +--------------+                   
                  | data/raw.dvc |                   
                  +--------------+                   
                          *                          
                          *                          
                          *                          
                      +------+                       
                      | load |                       
                    **+------+**                     
                 ***            ***                  
               **                  **                
             **                      **              
  +----------------+            +---------------+    
  | clean-features |            | clean-targets |    
  +----------------+            +---------------+    
           *                            *            
           *                            *            
           *                            *            
+--------------------+        +-------------------+  
| transform-features |        | transform-targets |  
+--------------------+        +-------------------+  
                 ***            ***                  
                    **        **                     
                      **    **                       
                     +-------+                       
                     | train |                       
                     +-------+    
```

To reproduce the results, run:

`dvc repro`

To show the resulting model metrics, run:

`dvc metrics schow`

## Using the API

#### Local Execution

To test the API locally, you can spin up the container by running:

`docker-compose up`

The documentation of the API can than be seen on `localhost:8000/docs`.

#### Live Endpoint

The API is hosted on [Heroku](https://heroku.com). The documentation of the live API can be seen on:

`https://fair-compensation.herokuapp.com/docs`

> **_NOTE:_**  The service is shut down automatically to save resources, when it's not used for some time.
> It might take some time to start the service again once you call the link.

## Additional Information

Please be aware that this project serves as a showcase.
For additional information, suggestions for improvement or collaboration feel free to contact [me](https://t.me/marcelfe).