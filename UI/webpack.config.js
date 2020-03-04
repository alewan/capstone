const path = require('path')
const webpack = require('webpack')

module.exports = {
   mode: 'production',
   entry: [
      'webpack-hot-middleware/client?path=/__webpack_hmr&timeout=20000',
      './src/app.js'
   ],
   output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'bundle.js',
      publicPath: '/static/'
   },
   externals: {
      react: 'react'
   },
   module: {
      rules: [
         {
            test: /\.js$/,
            // exclude: /node_modules/,
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
