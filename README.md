## WATER POTABILITY PREDICTION

Hello everyone, my goal in this project was to build a model with Water Quality and Potability dataset and deploy it. Our aim in the project is to find out whether the water is of good quality or poor quality through data. If the prediction is **1** , the result shows that the **drinkable(good quality)**, if it is **0**, it shows that the water is **undrinkable**.



* The technologies I used are as follows.

![image](https://github.com/user-attachments/assets/f6c9f43d-338a-4ce1-8fbe-b6dd0cb88879)


* The dataset I used:
[dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability)

* My notebook about the dataset: [notebook](https://www.kaggle.com/code/yusufcizlasmak1/catboost-eda-adasyn)

## Requirements
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)
- [Kubernetes](https://kubernetes.io/)
- [Docker](https://www.docker.com/)

##  Install packaging 


```
pip install -r requirements.txt
```

* Now you have already Pipfile Pipfile.lock files..
## Docker

```
docker image pull ycizlasmak/waterapp:1.2.1
```

* After creating your docker image, convert it to a container.
```
docker run -it -p 8000:8000 ycizlasmak/waterapp:1.2.1
```

* is it working ? Checking..
```
docker ps 
```

More information : 



## Results




https://github.com/user-attachments/assets/924a781b-4e71-408e-9940-e12f124f170d



