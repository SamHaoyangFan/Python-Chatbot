# Python-Chatbot
---

A Python chatbot using natural language processing and deep learning.

---
## How to run the app?

Please follow the following step inside your command prompt:

```python trainbot.py```

```python chatai.py```

## Approach
```trainbot.py``` will train the data from the local databse, and compile the model with SGD

```chatai.py``` set up the bot response, GUI, and several other functions (such as search function)

```intent.json``` stores the words/sentence that bot will be respond to, it is the database

## Chatbot AI
![image](https://user-images.githubusercontent.com/105527191/220520316-a690562a-1572-4e69-9a85-e93fee9211c9.png)

## Search Function
The bot can also search and locate the top 5 hyperlinks of the subjects you want to search by type ```find``` or ```search``` 
![image](https://user-images.githubusercontent.com/105527191/220520596-c71a3d31-204f-4307-8b05-2e1f535962b4.png)

## What is the RNN test?
The RNN test folder is an alternative way I used to build the chatbot.

The code is working, but it requires a larger database to run. The larger the database, the more accuracy with the RNN model

If you have a larger database for testing, replace ```trainbot.py``` and ```chatai.py``` with the files in RNN test folder
