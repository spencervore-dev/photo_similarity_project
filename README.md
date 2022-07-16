# PEXEL DOWNLOAD SCRIPT DOCUMENTATION

This script will download all the available images on pexel matching search criteria using the pexel api.


## Environment
This script was developed in Python 3.10.4. Install requirements.txt for 3rd party packages:
```pip install -r requirements.txt```


## Usage
Copy the file named config_template.py into config.py, add your secret API key, and adjust the settings. 
Ensure that config.py is not added to your git repo by running ```git status``` after you make the config.py
file so no secret keys get added to our repo.

Visit this website for more information on how to obtain a Pexel API key:
https://www.pexels.com/api/documentation/#authorization

Then, run the file pexel_api_pull.py to start the download based on the settings in config.py. All the images
matching your API query will be saved into whichever directory you defined as your download location.




