import urllib
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"remote control","limit":80,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)

for path in paths:
	urllib.urlretrieve(path, "00000001.jpg")
