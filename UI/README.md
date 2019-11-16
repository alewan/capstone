# Front End UI
2019-2020 Capstone Project

This directory holds the Front-end template for the Social Cue Detection Application. You can run the front end locally using the instructions below. 

## Directory Structure

### index.html
Contains front HTML script for UI (and currently also contains CSS, to be separated in future PR).

### server.js
Contains Node.js server code. Server listens on port 3000 when running, stores uploaded file to folder 'Uploads' and posts confirmation response to user.

### styles.css
Contains css specifications, currently not implemented.

## Installation
Use NPM (or equivalent package manager) to install dependenceis. Dependencies for this component are Node.js, Multer and Express. Ensure that you are in the UI directory when running installation commands.

### To install Node.js, follow the instructions for your machine:
https://nodejs.org/en/download/
 
### To install dependencies:
$ npm install multer
$ npm install express

## Usage
To run the front-end, cd into the UI directory. Create a new directory inside UI called "Uploads".
Then run:

$ node server.js

You should see "Working on port 3000" in terminal from which you are running the server.
Now, go to: http://localhost:3000/. You should see the front-end webpage.

Next, upload a file. Once the file has been successfully uplaoded, you should see it in the "Uploads" folder created earlier. If you didn't create this folder, you will see an error message. 