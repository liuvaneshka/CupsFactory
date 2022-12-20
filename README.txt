Anana Express


Run program with: python3 main.py
It must be first saved in TP_Arch_config folder
with the following files: https://drive.google.com/drive/folders/1WBb_Z5_vjlQrwfPJiukMXkgh6JeGjfYN
pedidos.csv


CALCULATE ROUTE

* Starting point: BUENOS AIRES.
* Orders deliveries are made in a circular way.
* Deliveries areas made by using the method abs of gps, latitude values are for indicate the delivery area
* A list stores the cities of the orders, we give the location to the method geolocator.geocode.
* A dictionaries stores the latitudes of each city [city would be the key] we order the dictionaries by latitude. We add each city to the tour list.


Artificial intelligence:

* We created the functions that would define the range necessary for the detection of a specific color.
* We load the configuration file and the weights file to build the network, we load the names of the classes.
* a BLOB is built from the image and we create an Input to the network.
* The class index with the highest confidence score is identified.
* We select the prediction boxes with a confidence of more than 30% to associate the color with the article we create a label variable that is created in     the detect image function.
* if the label is not "cup" or "bottle" we launch the stopped process message since In case it detects an animal the process "stops", in this function a      dictionary is created that will store the stock of each product based on the images of lot0001.
* From the available stock, it returns a simpler dictionary with the availability of colors
   of each item.
