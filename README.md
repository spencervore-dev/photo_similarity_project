# Image similarity project
The goal of this project is to take a collection of images and try to map them into a space where we can find other 
visually similar images.

## Project Sturcture
1) Pexels download stuff: pexels_api_pull.py, config_template.py - use these scripts to download up to 8000 images from
   www.pexels.com to get a dataset.
2) Pixabay download stuff: pixabay_api_pull.py, config_pixabay_template.py - These scripts are very similar to the Pexels
    scripts, except they were modified to run on Pixabay to try an alternate data source. Pixabay's API is limited to
    about 600 photos per search.
3) isomap: isomap_v1.py: To map the images into a space where the nearest neighbors are visually similar, we tried the
    isomap algorithm in this script.

## Environment
This script was developed in Python 3.10.4. Install requirements.txt for 3rd party packages:
```pip install -r requirements.txt```


## PEXEL DOWNLOAD SCRIPT DOCUMENTATION

This script will download all the available images on pexel matching search criteria using the pexel api.

To use, copy the file named config_template.py into config.py, add your secret API key, and adjust the settings. 
Ensure that config.py is not added to your git repo by running ```git status``` after you make the config.py
file so no secret keys get added to our repo.

Visit this website for more information on how to obtain a Pexel API key:
https://www.pexels.com/api/documentation/#authorization

Then, run the file pexel_api_pull.py to start the download based on the settings in config.py. All the images
matching your API query will be saved into whichever directory you defined as your download location.


### PIXABAY DOWNLOAD SCRIPT DOCUMENTATION
Very similar to the design of teh pexels script. Copy the file named config_pixabay_template.py to config_pixabay.py
and add your Pixabay API key, which you can get by setting up an account on pixabay.com.

Pixabay API documentation is here:
https://pixabay.com/api/docs/

Run the file pixabay_api_pull.py to start pulling data from pixabay based on the settings in your config_pixabay.py file.


