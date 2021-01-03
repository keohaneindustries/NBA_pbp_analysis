# NBA Play-By-Play Data Analysis

## Quick Start Guide

1. (optional) for better read/write performance, copy the entire directory of raw data off the USB 2.0 stick to its long-term home.  Probably want it on a USB 3.0 stick or directly on your a hard drive.
2. clone this repo to your local machine (recommend working off of a branch, obviously).
3. edit the directories and filepaths hard-coded in `.nba_pbp_analysis/data/cf.py` to match the locations on your local machine.  
4. compile and store a relatively clean, uniform dataset from the various raw data sources by running `.nba_pbp_analysis/data/pbp/cleaning/clean.py`.  
5. confirm the data cleaning was successful and view some samples of the data by running `.nba_pbp_analysis/data/load.py`.

Note: This repo is currently NOT a library that you can install to your venv. I recommend working out of PyCharm.


## Data Source Overview

### Eight Thirty Four

[link to 834 website](https://eightthirtyfour.com/data)

* was free
* has a small amount of documentation online
* the guy who made the website says DM him on twitter if interested

To-Do List  
* figure out what numerical event codes mean (no definition online) and write some code to map event codes to text (e.g. EVTMSGTYPE==1 -> made FG).  
* find 2019-2020 data.  
* contains a lot of redundant data (e.g. columns for both team name and team abbreviation), so select the data points you want and add logic to the cleaner to drop the others.  


### BigDataBall

[link to BigDataBall website](https://www.bigdataball.com/datasets/nba/)

* was not free
* zero documentation online
* has a handful of additional datapoints
* much more tedious to work with b/c not parsed out as cleanly

To-Do List
* rows generally only contain identifiers for one of the two teams playing (smdh). You'll need to create a totally separate data preparation step where you do something like group by game ID and deduce the home/away teams by mapping events to teams to players to home/away.
