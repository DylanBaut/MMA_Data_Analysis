# PythonScraping
Python web scraping for UFC website stats (implement multithreading, multiprocessing, asynchronous)
Single page of events:
Without multithreading/multiprocessing: 196.16 seconds (3.2 min)
With Multiprocessing: 13.35-41 seconds
With Multithreading: 23 - 40 seconds
With Asynchronous: 18.71 - 39 seconds
Spreads differ wildly, however
(limited by 429 TOO MANY REQUESTS Error on UFC Stats server side)

Conclusion: Minimum ceiling of under 40 seconds for each page of 24 events