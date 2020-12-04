# Scope of Data Science project

## Background
According to the Zindi Data Science competition, road traffic collisions are the number one killer of children and young adults ages 5-29, and 8th leading cause of death worldwide. 
Post-crash care is one of the five pillars of road safety and a critical component for reducing morbidity and mortality.
When it comes to emergency response to road accidents, every second counts. With heavy traffic patterns and the unique layout of the city, 
finding the best locations to position emergency responders throughout the day as they wait to be called is critical in a city like Nairobi.

## Goal
Using historical crash data, weather data and road segment data, the goal is to improve the response time to road traffic accident (RTA) sites of a set
of 6 ambulances in the city of Nairobi.
If time permits, appealing visualizations would be welcome as well.

## Business Use Case
We aim to achieve this goal by creating a python script that supports decision makers who coordinate ambulances. The script will involve a module
that predicts future road traffic accident (RTA) locations and times based on historical crash site data and additional environmental empirical data. This will serve as the basis for another module which determines an optimized allocation of ambulances to locations in order to decrease the overall emergency response time.
Starting point will be a baseline model that assumed euclidean distances between ambulance location and crash site - this will then be extended to consider real 
travelling distance and later be converted to travelling time. Finally we want to create a model which minimizes the number of responses that exceed 
a certain "golden" threshold value.
The allocation algorithm for the ambulances should provide updated optimal locations in case one or many ambulances are committed to an RTA or otherwise
unavailable.

## Resources available
Available data:
* Historical data on RTA for the time frame of 01/2018 - 07/2019. Consists of time of the accident and latitute and longitude.
* Historical Weather data for the same time frame. Consists of one entry per day and is composed of 6 different features like precipation or humidity.
* Road segment data for some of the crash sites. Consists of the segment name and 227 obscured features.

## Deliverables
Deliverable   | Description
--------------|-------------
Deliverable 1 | Good position on leaderboard
Deliverable 2 | Placement function API
Deliverable 3 | Presentation of results and tool
Deliverable 4 | Interactive visualization

## Project timeline
Deadline | Milestone or tentative goal
---------|----------------------------
Week 1 | <ul> <li>Organisational setup</li> <li>Data exploration</li> <li>Submission to Zindi</li> </ul>
Week 2 | <ul> <li>Clustering algorithms implemented</li> <li>Regression algorithms implemented</li> </ul>
Week 3 | <ul> <li>End-to-end pipeline</li> <li>Minimum viable product</li> <li>Expanded input data set</li> </ul>
Week 4 | <ul> <li>Automated and optimized pipeline</li> <li>Visualization</li> <li>Presentation</li> </ul>
