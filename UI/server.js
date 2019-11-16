var express = require("express");
var multer = require('multer');
var app = express();

var storage = multer.diskStorage({
  destination: function (req, file, callback) {
    callback(null, './uploads');
  },
  filename: function (req, file, callback) {
    callback(null, file.fieldname + '-' + Date.now() + '.mp4');
  }
});

var upload = multer({ storage : storage}).array('userPhoto');

app.get('/', function(req, res){
      res.sendFile(__dirname + "/index.html");
});

// get file from user
app.post('/api/photo', function(req, res){
    upload(req, res, function(err) {
        if(err) {
            return res.end("Error uploading file.");
        }
        res.end("Emotion Classification: y x s");
    });
});

// app.use(express.static(path.join(__dirname,"public")));

app.listen(3000, function(){
    console.log("Working on port 3000");
});