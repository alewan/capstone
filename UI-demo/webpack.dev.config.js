const path = require('path')
const webpack = require('webpack')

module.exports = {
   mode: 'development',
   // node: {
   //    fs: 'empty'
   //  },
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
         }, 
         // {
         //    test: /\.(ttf|eot|svg|gif|jpg)(\?v=[0-9]\.[0-9]\.[0-9])?$/,
         //    include: /node_modules/,
         //    use: [{
         //        loader: 'file-loader'
         //    }]
         // }
         {
            test: /\.(png|jpe?g|gif)$/i,
            use: [
              {
                loader: 'file-loader',
              },
            ],
          },
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

