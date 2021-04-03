# Virtual-Eyeglass-Flask

1. You have to install virtual environment to run this project.
2. To run project on local, please run "flask run"
3. You can check project on localhost:5000

Note:
```
To deploy Flask Applicatoin to Heroku please refer this URL:
[https://stackabuse.com/deploying-a-flask-application-to-heroku]

When deploying to Heroku.com you have to refer below two points.

1) Tensorflow 2.0 module is very large because of its GPU support. Since Heroku doesn't support GPU, it doesn't make sense to install the module with GPU support.
Solution:
Simply replace tensorflow with tensorflow-cpu in your requirements.
Refer URL: [https://stackoverflow.com/questions/61796196/heroku-tensorflow-2-2-1-too-large-for-deployment]

2) Heroku doesnot support libSM6 dependency. So you have to use opencv-python-headless instead of using opencv-python.
Refer URL: [https://stackoverflow.com/questions/49469764/how-to-use-opencv-with-heroku]
```
