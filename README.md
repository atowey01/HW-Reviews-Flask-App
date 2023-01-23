# HW-Reviews

**Project Description**

HW-Reviews aims to analyse Hostelworld reviews in more detail to gain a better understanding of the hostel before visiting. Sometimes hostel ratings seem higher than visitors experience of the hostel so HW-Reviews helps to identify any inaccuracies there may be.

Insights include:
- Providing a predicted rating of a hostel solely based on the sentiment of the text previous visitors have left for the hostel
- Displaying positive and negative sentences in reviews of a hostel clearly
- Showing some reviews where the rating the customer provided and the predicted rating based on the text they wrote may be different

The app was created using Python and deployed using Flask on Heroku. There are two machine learning models used, one to predict the sentiment of sentences in hostel reviews and another to predict the rating of a review solely based on the text reviews previous visitors have left. 

**How to Install and Run the Project**

1. Clone the repository
2. Ensure you are using python-3.9.16
3. Install the requirements.txt file
4. If running the project locally you will need to add the following code to the bottom of the app.py file:
    '''
    if __name__ == '__main__':
       app.run(debug=True, port=8080)
    '''
5. Run the app.py file using the command "python3 app.py"

Note - Heroku uses the Procfile to run the app using gunicorn

**More Project Information**

The model_training_scripts directory provides files and instructions for both scraping data to create labels to train models (.py files) and the Jupyter notebooks used to train both models (.ipynb) files. To data used to train the models is not included in this repo due to the file size. Contact the repository owner to get the data or use the scripts to make your own dataset. Very few labels were used to train these models to date. Both models could be improved with more labelling.

The random forest rating prediction model is contained in this repo. The sentiment model is downloaded from the HuggingFace model hub.

If the repository grows more the Flask code should be changed to include blueprints to improve readability.