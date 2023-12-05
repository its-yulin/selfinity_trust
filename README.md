# Selfinity_Trust: Personal Database
### Author: Yulin Hu, Yijun Liu

<img width="958" alt="demo screenshot" src="https://github.com/its-yulin/selfinity_trust/assets/91909405/7df1acda-3ca8-4e3f-b6c9-4fce59a738f1">

## 🎉New features
This adapted version aims to improve user trust and model reliability by...
- Differentiate user command types
- Run logic check
- Provide downloadable evidence (raw data)

## 💎 What it does
Our AI assistant serves as a comprehensive digital life co-pilot, redefining the way we interact with our digital lives. Its capabilities go beyond mere data consolidation; it actively assists in managing and interpreting information. Check out the original version here: https://github.com/its-yulin/Selfinity_AI

## 💻 Brief Code Documentation
The main backend function is located in the [app.py](/app.py). It contains all necessary functions for this project. 
Specifically for this course, the functions we improved are in the generate() function and its auxiliary functions.
The frontend files are located in two separate folders. The [templates](./templates) have the HTML skeletons. The [static](./static) has the CSS as well as JS files for the front end. 
To run this project locally, please 1) download the project 2) install the packages in requirements.txt 3) insert your PINECONE_API_KEY, OPENAI_API_KEY, and PLAID_CLIENT_ID 4) run app.py file. 

