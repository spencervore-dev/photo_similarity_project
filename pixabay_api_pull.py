# This script bulk downloads photos from the Pexel API and saves them into a folder

import requests
from pprint import pprint
import json
import time
import traceback
import re
import shutil
from datetime import datetime
import time
import random as rand

import config_pixabay as c


# define cache - set to 24 hours
import requests_cache
requests_cache.install_cache(cache_name='pixabay_cache', backend='sqlite', expire_after=24*60*60)


# Some variables to track progress of script
total_images_downloaded = 0
script_starttime = datetime.now()
print(f"Pixabay API image download script has started at {script_starttime}")

def api_call(api_key, url, verbose=True):
    '''
    make api request to pexel and get response

    api_key: Your pexel api key as a string
    url: url to use for get request
    verbose: Boolean. If True, print debugging output.

    '''

    # Continue looping until API request works, in case there's 
    # network problems... the pexel website goes down, etc
    success = False
    while not success:
        try:
            response = requests.get(url)
            response.raise_for_status()
            success = True
        except:
            if response._content == b'[ERROR 400] "page" is out of valid range.':
                return {"terminate": True}
            success = False
            print("WARNING: API request failed... retrying")
            try: print(response.status_code, response.reason)
            except: pass
            print(traceback.format_exc())
            time.sleep(10) # Wait 10 seconds if call fails to let things cool off...

    data = json.loads(response.text)

    # print responses
    if verbose:
        print(response.status_code, response.reason)
        pprint(data)


    return data


def format_string(string, truncate_val = 40):
    '''used to format response information'''
    string = string.strip() # remove leading and trailing whitespace
    string = re.sub(r"-", '_', string) # replace - with _ to make more linux friendly
    string = re.sub(r"\s+", '_', string) # replace whitespace with _ to make more linux friendly
    string = re.sub(r"[^\w\s_]", '', string) # Remove all non alphanumeric or _ chars
    string = string[0:truncate_val] # Truncate string after certain number of chars to avoid filename too long errors later.
    return string


def download_images(data, filepath_destination, page = None, search_term = "", verbose = True):
    '''
    This function takes a list of imaage urls in the json returned by the pixabay api, and then downloads them.

    data: JSON as dict form pixabay search api. This contains a list of image url's to download
    filepath_destination:  Directory on your local system where you want to save all these images to.
        Don't use filepath shortcuts like ~
    page - API results page number, used for labeling saved images
    search_term - Search term used in pixabay api, used for labeling
    verbose: If true, print extra console output
    '''

    images_downloaded = 0 # Track images downloaded only for this function call
    photos = data['hits']
    for index, photo_json in enumerate(photos):

        # Parse JSON for relevent information and construct output filename
        img_id = photo_json['id']
        img_url = photo_json['webformatURL']
        title = photo_json['pageURL'].strip().split('/')[-2]
        if verbose:
            print(photo_json['pageURL'])
            print(title)
        title = format_string(title)
        if verbose: print(title)

        photographer = photo_json['user']
        photographer = format_string(photographer)

        photographer_id = photo_json['user_id']
        
        # Idea for extracting filename from URL from:
        # https://towardsdatascience.com/how-to-download-an-image-using-python-38a75cfa21c
        img_filename_on_web = img_url.split("/")[-1].split("?")[0]
        img_ext = img_filename_on_web.split(".")[-1]
        if verbose: 
            print(img_url)

        filename = f"pixabay_image__pg{str(page).zfill(4)}_ind{str(index).zfill(3)}_" + \
                    f"imgid{str(img_id)}_{search_term}__" + \
                    f"{title}_by_{photographer}_authorid{str(photographer_id)}.{img_ext}"

        if verbose: 
            print(f"\n\nData for photo index {index} of page {page}:")
            print(f"Image title is: {filename}\n")
            pprint(photo_json)


        # Download Image URL in Python
        # Methods from: https://towardsdatascience.com/how-to-download-an-image-using-python-38a75cfa21c
        # img_url = "http://www.spencervore.com"  # Test broken link error handling
        success = False
        while not success:
            try:
                img_response = requests.get(img_url, stream = True)
                img_response.raise_for_status()
                img_response.raw.decode_content = True
                with open(filepath_destination + filename, 'wb') as f:
                    shutil.copyfileobj(img_response.raw, f)
                success = True
                images_downloaded += 1
            except:
                success = False
                print("WARNING: Downloading image from url request failed... retrying")
                try: print(img_response.status_code, img_response.reason)
                except: pass
                print(traceback.format_exc())
                time.sleep(10) # Wait 10 seconds if call fails to let things cool off...

        # Slow down the request rate a bit, if needed
        time.sleep(image_pause_time)
    return images_downloaded


# Execute main body
# ------------------------------------------

# Import configuration
config_pixabay_api_key = str(c.pixabay_api_key)
search_term = str(c.search_term)
color = str(c.color)
per_page = int(c.per_page)
url = str(c.url)
filepath_destination = str(c.filepath_destination)
max_api_calls = int(c.max_api_calls)
verbose = bool(c.verbose)
image_pause_time = int(c.image_pause_time)
page_pause_time = int(c.page_pause_time)

# Just keep looping over api calls
calls = 1
while calls <= max_api_calls:

    # Get list of image URLs as Json from pexel API
    print(f"Fetching page {calls}. Each page contains {per_page} image results...")
    url_iter = url
    if calls > 1:
        url_iter = url + f"&page={calls}"
    if verbose: print(url_iter)
    data = api_call(config_pixabay_api_key, url_iter, verbose=verbose)
    if "terminate" in data and data["terminate"] == True:
        print("No next page of results... end of results reached")
        "This is the returned response code when no more results are retrieved"
        break

    if verbose: print(data.keys())

    if calls == 1:
        print(f"TOTAL NUMBER OF AVAILABLE IMAGES IS: {data['totalHits']}\n\n")

    # Parse image list json and download each image url to a folder
    total_images_downloaded += download_images(data, filepath_destination,
            page=calls, search_term = search_term, verbose = verbose)

    print("Taking a short break")
    time.sleep(page_pause_time)

    '''
    if 'next_page' in data:
        url = data['next_page'] # Use link to next page of results to keep querying
    else:
        print("No next page of results... end of results reached")
        break
    '''

    calls += 1

# Finish script - display summary
script_endtime = datetime.now()
print("\n\n------------------------------------------")
print("PIXABAY API IMAGE DOWNLOAD SCRIPT EXECUTED SUCCESSFULLY")
print(f"{total_images_downloaded} images were downloaded")
print(f"Images were saved to {filepath_destination}")
print(f"Script execution ended at {script_endtime}")
print(f"Script execution took {script_endtime - script_starttime} (HH:MM:SS:XXX)")
print()



