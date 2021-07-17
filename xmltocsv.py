import xml.etree.cElementTree as ET
import pandas as pd

def xmltocsv():
    path = "11.xml"
    tree = ET.parse(path)
    root = tree.getroot()

    classes = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for element in root.findall("filename"):
        filename = element.text
        
    for element in root.findall("size"):
        width = element.find("width").text
        height = element.find("height").text

    for element in root.findall("object"): 
        category = element.find("name").text
        classes.append(category)
        xmi = (element.find("bndbox")).find("xmin").text
        xmin.append(xmi)

        ymi = (element.find("bndbox")).find("ymin").text
        ymin.append(ymi)

        xma = (element.find("bndbox")).find("xmax").text
        xmax.append(xma)

        yma = (element.find("bndbox")).find("ymax").text
        ymax.append(yma)

    # dictionary of lists  
    content = {"Filename" : filename, "Width" : width, "Height" : height, 'Class' : classes, "Xmin" : xmin, "Ymin" : ymin, "Xmax" : xmax, "Ymax" : ymax}  
     
    df = pd.DataFrame(content) 
  
    # saving the dataframe 
    df.to_csv('Annotations.csv')

xmltocsv()  