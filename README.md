# MMA Data Analysis

WEB APP LINK: 
https://mma-round-judge.herokuapp.com/

#Demo:
Round Judge
![image](https://github.com/DylanBaut/MMA_Data_Analysis/assets/70183425/5ff869fa-91e5-42c3-a7b9-2321ff97374b)

Fight Scorer
<img width="1615" alt="Screen Shot 2023-07-06 at 3 47 41 PM" src="https://github.com/DylanBaut/MMA_Data_Analysis/assets/70183425/f8605bd8-a4b2-4baa-9ab5-72adee33c870">

<img width="1618" alt="Screen Shot 2023-07-06 at 3 47 18 PM" src="https://github.com/DylanBaut/MMA_Data_Analysis/assets/70183425/7978b093-9380-4c76-8a69-4adc6c784d69">

'''One of the most contentious and seemingly arbitrary aspects of the sport of mixed martial arts is judging decisions. If you're not familiar, every fight is either 3 or 5 rounds, and if a finish isnt achieved within the time limit, 3 judges will have judged each individual round. Each round is based off of a 10 point must system, awarding the chosen winner 10 and the loser less than 10.

Judging has been critisized by fans and fighters alike due to the vagueness of how much factors like damage, control time, grappling, octogon control, etc. are actually valued in an assessment of a round. The goal of this project is to analyze historical data across all UFC rounds, in addition to fan/media agreement on decisions. 

Created is python code that web srapes two sites: mmadecisions.com (for decisions/ fan and media scoring) and UFCstats.com (for round by round statistics). These scripts utilizes microprocessing to optimize the time it takes to gether all the information. Then, the combined csv files are converted to a 103x2306 dataframe. This data will be used in a nueral network and other predictive alogorithms.
'''

Python web scraping for UFC website stats 

Notes:
(implement multithreading, multiprocessing, asynchronous)
Single page of events:
Without multithreading/multiprocessing: 196.16 seconds (3.2 min)
With Multiprocessing: 13.35-41 seconds
With Multithreading: 23 - 40 seconds
Main.py scrapes data from every 5/3 round fight in UFC history:
With Asynchronous: 18.71 - 39 seconds
Spreads differ wildly, however
(limited by 429 TOO MANY REQUESTS Error on UFC Stats server side)
Conclusion: Minimum ceiling of under 40 seconds for each page of 24 events

Final Result:
All UFC data across history scraped within 1002.64 seconds (16.7 minutes)
Makes sense since (1002.64/about 26 pages) = 38.5 seconds per "page"

Decisions.py scrapes data from every 5/3 round fight in UFC history with available scorecords for all 3 judges:

Combine.py appends every decision data row with its corresponding row in UFC stats. (Some names that dont match betwen the two sites were exclude)
Combine.py operates in 224.40 seconds.
#Must download requierements.txt
'pip install -r requirements.txt'

Neural Network:
Accuracy 0.8200549483299255 Linear(7, 12), ReLU(), Linear(12, 7), ReLU(), Linear(7, 7), CrossEntropyLoss(), optim.Adam(lr=0.001)

Web App structure:
- Welcome page,two options:
    -  judge-trained model
    -  judge/fan-traiend model
- work page:
    - enter 14 stats in field boxes: gived percentages of each scoring outcome, with most likely highlighted in center
    - Allow for search feature of "Was this a robbery?"
        - search option for every single UFC fight, show predicted scores for each round (with percentages)
        - shows if robbery or not
