var path = require('path')
var multer = require('multer')
var express = require('express')

// webpack
var webpack = require('webpack')
var webpackDevMiddleware = require('webpack-dev-middleware')
var webpackHotMiddleware = require('webpack-hot-middleware')
var config = require('./webpack.dev.config')
var compiler = webpack(config)

var app = express()

var pipelineScript = '../pipeline/pipeline.py';
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
      console.log(results[0]);
      res.status(200).json({
         "result":results[0]
      })
   });
});

app.listen(8080, function () {
   console.log(`Starting capstone emo app. on port 8080`)
})
