## ML Project

Hello everyone, my goal in this project was to build a model with Water Quality and Potability dataset and deploy it. Our aim in the project is to find out whether the water is of good quality or poor quality through data. If the prediction is **1** , the result shows that the **drinkable(good quality)**, if it is **0**, it shows that the water is **undrinkable**.



* The technologies I used are as follows.
![image](https://github.com/Yusuf-Cizlasmak/End_to_End_ML_Project/assets/97342455/f0da38c3-2089-4102-a7f6-71edd2c5d67d)


* The dataset I used:
[dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability)

* My notebook about the dataset: [notebook](https://www.kaggle.com/code/yusufcizlasmak1/catboost-eda-adasyn)

## Requirements

- [Uvicorn](https://www.uvicorn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://www.docker.com/)

##  Install packaging 
```
pip install -r requirements
```


## Docker

```
docker build -t fastapi .
```

* After creating your docker image, convert it to a container.
```
docker run -it -p 8000:8000 fastapi
```

* is it working ? Checking..
```
docker ps 
```


## Results

![ezgif com-video-to-gif(1)](https://github.com/Yusuf-Cizlasmak/End_to_End_ML_Project/assets/97342455/4cfdd68d-83c1-4542-bad5-c5eed767baf1)


