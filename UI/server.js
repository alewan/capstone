var express = require('express')
var fs = require('fs')
var multer = require('multer')
var path = require('path')

var webpack = require('webpack')
var webpackDevMiddleware = require('webpack-dev-middleware')
var webpackHotMiddleware = require('webpack-hot-middleware')
var config = require('./webpack.dev.config')
var compiler = webpack(config)

var app = express()

var pipelineScript = '../pipeline/pipeline.py';
const {spawn} = require('child_process');

const {PythonShell} = require("python-shell");

app.use(webpackDevMiddleware(compiler, {
   publicPath: config.output.publicPath,
   noInfo: true,
   quiet: false,
   historyApiFallback: true,
   stats: {
      colors: true
   }
}))

app.use(webpackHotMiddleware(compiler, {
   log: console.log,
   path: '/__webpack_hmr',
   heartbeat: 10 * 1000
}))

app.use('/assets', express.static('assets'))

// file upload:
var storage = multer.diskStorage({
     destination: function (req, file, callback) {
       callback(null, '../pipeline/pipeline_helper/user_upload');
     },
     filename: function (req, file, callback) {
       callback(null, 'input_file.mp4');
     }
   });

var upload = multer({ storage : storage });

app.get('/', function(req, res){
   res.sendFile(path.join(__dirname + "/index.html"));
});

app.post('/files', upload.any(), function (req, res, next) {
   req.files.forEach((file) => {
      console.log(file)
   })
   res.status(200).end()
})

// classify emotion:
app.post('/classify', function(req, res){

   console.log("running pipeline to classify .mp4...")
  
   var classification = PythonShell.run(pipelineScript, null, function (err, results) {
      if (err) throw err;
      //fs.readFile("AWS_Results.json", function(err, buf) {          
        //let data = JSON.parse(buf);
        //let emotions = data[0][1].FaceDetails[0].Emotions
        //var UI_results = JSON.stringify(emotions);
        //res.end("results: " + UI_results);
        //res.render({UI_results: UI_results});
      console.log(results);
      res.status(204).send();
   });
});


app.listen(8080, function () {
   console.log(`Starting emo app. on port 8080`)
})
