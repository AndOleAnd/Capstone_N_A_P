# Capstone_Nairobi_Ambulance_Challenge

Repository for working on NF Capstone Project



## Challenge Details

https://zindi.africa/competitions/uber-nairobi-ambulance-perambulation-challenge



## Setup

The input data is in the *Inputs* folder. This folder is not synchronized with GitHub. Produced outputs will go into the *Outputs* folder which is synchronized.

The conda environment *nairobi_ambulance* contains:
* geopandas 0.8.1 - use pip install
* h3-py 3.7.0
* holidays 0.10.3
* jupyter 1.0.0
* jupyterlab 2.2.9
* pandas 1.1.3
* scikit-learn 0.23.2
* seaborn 0.11.0



## Collecting data

Suggested datasets:
1. Training data - recorded crashes from 2018-01-01 to 2019-06-01 (Lat,Lon and time)
2. Road survey data from Uber Movement
3. Weather pattern data
4. Segment data - Detailed but unlabelled road characteristic info. 

Some insights into the problem:
https://kenya.ai/ai-kenya-ambulance-perambulation-datathon-highlights/

Intro to the problem from Flare:
https://www.youtube.com/watch?v=YjV367g-0sA&feature=emb_logo

Baysean Approach to the problem:
https://www.youtube.com/watch?v=66IhpZ8p2y0&feature=emb_logo



Training data:



Getting Uber movement data via commandline
* https://www.npmjs.com/package/movement-data-toolkit
  * in case node.js need to be installed:  
  ```npm install -g npm```
* Once that is installed:  
    ```npm install -g movement-data-toolkit```



