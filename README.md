# PythonScraping
Python web scraping for UFC website stats (implement multithreading, multiprocessing, asynchronous)
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