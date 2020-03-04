const path = require('path')
const webpack = require('webpack')

module.exports = {
   mode: 'development',
   entry: [
      'webpack-hot-middleware/client?path=/__webpack_hmr&timeout=20000',
      './src/app.js'
   ],
   output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'bundle.js',
      publicPath: '/static/'
   },
   module: {
      rules: [
         {
            test: /\.less$/,
            //exclude: /node_modules/,
            use: ['url-loader']
         },
         {
            test: /\.js$/,
            exclude: /node_modules/,
            use: ['babel-loader']
         }
      ]
   },
   plugins: [
      new webpack.HotModuleReplacementPlugin()
   ],
   resolve: {
      extensions: ['.js'],
      enforceExtension: false
   }
}

