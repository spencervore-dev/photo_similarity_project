# CONFIG FILE
# Use this file to define variables that adjust the script run configuration
# The script will only work if this file is named config.py. Rename this template to config.py after you add 
# your api key and adjust configuration. Ensure config.py is in your .gitignore file so you don't push your
# keys into your repository.


# SECRET CONFIGURATION
# DO NOT PUSH THESE VALUES INTO THE CODE REPOSITORY
pexel_api_key="<your api key here>" # Pexel API Key as string 
# Note: You will need to create a pexel account, then you can generate an API key. More details here:
# https://www.pexels.com/api/documentation/#authorization



# NON-SECRET CONFIGURATION
# Settings to refine api call. See API documentation here:
# https://www.pexels.com/api/documentation/#photos-search
search_term="animals" # Return photos that match this search term
size = "medium" # Size of photo to return
# orientation="portrait"
color = "purple" # Average color of photo to return
per_page = 80 # max allowable results per page is 80. This is how many photos you get per API request. Probably want it set to max.

# This is your query URL built from the previous settings
url = f"https://api.pexels.com/v1/search?per_page={per_page}&query={search_term}&color={color}&size={size}"  # &orientation={orientation}"

# Settings to save output - Where you want your bulk image download saved
# Note: You need the full filepath for this to work... don't use ~
filepath_destination = "/home/spencervore/Insync/OneDrive/isye6740_project_data/pexel/purple_medium/"

# Do not make more than this number of API calls
max_api_calls = 500

# If True, extra debugging output will print to your console
verbose = True
