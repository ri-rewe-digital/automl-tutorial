# AutoML - Tutorial #
Example on how to use a trained Google AutoML vision classification models in Python. 
AutoML is not a Free and in Beta at the time of the repository creation.
The provided model is a simple cat-dog-classifier which was created with an public Kaggle [Dataset](https://www.kaggle.com/tongpython/cat-and-dog#training_set.zip).

### Install ###
You need to have a running python3 and pip environment. With 
```bash
pip3 install -r requirements.txt
```
you can install the required python packages. If you have a running GPU setup with Nvidia driver, CUDA and CUDNN you can also install _tensorflow-gpu_.
This is optional and should only be done when you need the setup for your other ML projects too.

### Start ###
```bash
python3 prediction_controller.py
```
Now you have an rest application with the endpoints with _Content-Type=application/json_
```
POST@ localhost:5002/api/predict
GET@ localhost:5002/api/labels
```

The request body for the POST request looks like this
```json
{
    "identifier":"cat-dog-example-01",
    "prediction_results": 1,
    "base64Image":"base64encodedImage"
}
```
You can now use a REST client like Postman or curl to send the request
```bash
curl -H "Content-Type: application/json" -X POST -d @example-request.json http://localhost:5002/api/predict
```
The _prediction_results_ parameter is optional. If it is missing, you receive all labels. The response looks like the follows
```json
{
    "identifier": "cat-dog-example-01",
    "labels": [
        "dog"
    ],
    "scores": [
        0.8664201498031616
    ]
}
```
If you want multiple labels, they are returned order with with the highest score first.   
The labels endpoint returns a list of all available labels.
```json
{
    "labels": [
        "cat",
        "dog"
    ]
}
```