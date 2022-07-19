# CONFIG FILE
# Use this file to define variables that adjust the script run configuration
# The script will only work if this file is named config_pixabay.py. Rename this template to config_pixabay.py after you add 
# your api key and adjust configuration. Ensure config_pixabay.py is in your .gitignore file so you don't push your
# keys into your repository.


# SECRET CONFIGURATION
# DO NOT PUSH THESE VALUES INTO THE CODE REPOSITORY
pixabay_api_key="<your pixabay api key>" # Pixabay API Key as string 
# Note: You will need to create a pixabay account, then you can generate an API key. More details here:
# https://wpautocontent.com/support/knowledgebase/how-to-get-your-pixabay-api-key/ 



# NON-SECRET CONFIGURATION
# Settings to refine api call. See API documentation here:
# https://pixabay.com/api/docs/
search_term="animals" # Return photos that match this search term
color = "green" # Average color of photo to return
per_page = 10 # max allowable results per page is 200. This is how many photos you get per API request. Probably want it set to max.

# This is your query URL built from the previous settings
url = f"https://pixabay.com/api/?key={pixabay_api_key}&q={search_term}&image_type=photo&per_page={per_page}&safesearch=true&colors={color}&lang=en"

# Settings to save output - Where you want your bulk image download saved
# Note: You need the full filepath for this to work... don't use ~
filepath_destination = "/home/spencervore/OneDrive/isye6740_project_data/pixabay/test2/"

# Do not make more than this number of API calls
max_api_calls = 10

# If True, extra debugging output will print to your console
verbose = True


# Set some pause times that can be used to slow down the request rate if needed
page_pause_time = 1 # Number of seconds to wait between each page of image link results
image_pause_time = 0.5 # Number of seconds to wait between downloading each image
