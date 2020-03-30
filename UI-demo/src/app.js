import axios from 'axios'
import Blob from 'blob'
import FormData from 'form-data'
import React from 'react'
import ReactDOM from 'react-dom'
import Files from './file'

import preprocessed_img from '../../pipeline/pipeline_helper/images_preprocessed/input_file_0001.jpg';
import preprocessed_img_audio from       '../../pipeline/pipeline_helper/audio_preprocessed/input_file_rgb_plt_0001.png';

class Application extends React.Component {
  constructor (props) {
    super(props)
    this.state = {
      files: [], 
      classifiction: "",
      aws_results: [],
      audio_nn_results: [],
      final_predictions: [],
      pre_processed_image: [],
      pre_processed_audio_image: []
    }
  }

  onFilesChange = (files) => {
    this.setState({
      files
    }, () => {
      console.log(this.state.files)
    })
  }

  onFilesError = (error, file) => {
    console.log('error code ' + error.code + ': ' + error.message)
  }

  filesRemoveOne = (file) => {
    this.refs.files.removeFile(file)
  }

  filesUpload = () => {
    const formData = new FormData()
    Object.keys(this.state.files).forEach((key) => {
      const file = this.state.files[key]
      formData.append(key, new Blob([file], { type: file.type }), file.name || 'file')
    })

    axios.post(`/files`, formData)
    .then(response => window.alert(`File uploaded succesfully!`))
    .catch(err => window.alert('Error uploading file :('))
  }

  classifyEmotion = () => {
    console.log("running pipeline")
    console.log(this.state.classifiction)

    axios.post('/classify')
    .then(response => this.setState({classifiction: " Final Emotion Prediction: " + response.data.result}))
    .then(console.log(this.state.classifiction))
    .then(this.displayPipeline)
  }

  displayPipeline = () => {

    var AWS_RAVDESS_LIST = ['NEUTRAL', 'CALM', 'HAPPY', 'SAD', 'ANGRY', 'FEAR', 'DISGUSTED', 'SURPRISED']

    this.setState({pre_processed_image: [
      'Pre-Processed Image',
      preprocessed_img,
      ]
    })

    this.setState({pre_processed_audio_image: [
      'Pre-Processed Mel Specreal Image',
      preprocessed_img_audio,
      ]
    })

    // ************************************************************************** 
    // ******************************** AWS DATA ********************************
    // ************************************************************************** 

    // load, sort and set the data
    var data_aws = require('../../pipeline/pipeline_helper/aws_results.json')
    console.log(data_aws[0][1].FaceDetails[0].Emotions)

    var aws_preds = [0, 0, 0, 0, 0, 0, 0, 0]

    for (var i = 0; i <= 7; i++) {
      aws_preds[i] = {
          type: data_aws[0][1].FaceDetails[0].Emotions[i].Type, 
          confidence: data_aws[0][1].FaceDetails[0].Emotions[i].Confidence.toFixed(2)
        }
    }

    aws_preds.sort((a, b) => (a.confidence) - (b.confidence)).reverse()

    console.log("sorted aws predictions: " + aws_preds)

    this.setState({aws_results: [
      "AWS Predictions: ",
      aws_preds[0].type + ": ", aws_preds[0].confidence + "% ",
      aws_preds[1].type + ": ", aws_preds[1].confidence + "% ",
      aws_preds[2].type + ": ", aws_preds[2].confidence + "% ",
      aws_preds[3].type + ": ", aws_preds[3].confidence + "% ",
      aws_preds[4].type + ": ", aws_preds[4].confidence + "% ",
      //aws_preds[5].type + ": ", aws_preds[5].confidence + "% ",
      //aws_preds[6].type + ": ", aws_preds[6].confidence + "% ",
      //aws_preds[7].type + ": ", aws_preds[7].confidence + "% ",
      ]
    })

    // *************************************************************************** 
    // ************************ AUDIO NEURAL NETWORK DATA ************************
    // *************************************************************************** 

    // load, sort and set the data
    var data_nn = require('../../pipeline/pipeline_helper/audio_nn_predictions.json')
    console.log("audio pred: " + data_nn)

    var audio_nn_preds = [0, 0, 0, 0, 0, 0, 0, 0]

    for (var i = 0; i <= 7; i++) {
      audio_nn_preds[i] = {
          type: AWS_RAVDESS_LIST[i], 
          confidence: data_nn[i].toFixed(2)
        }
    }

    audio_nn_preds.sort((a, b) => (a.confidence) - (b.confidence)).reverse()

    console.log("sorted audio nn predictions: " + audio_nn_preds)

    this.setState({audio_nn_results: [
      "Audio Neural Net Predictions: ",
      audio_nn_preds[0].type + ": ", audio_nn_preds[0].confidence + "% ",
      audio_nn_preds[1].type + ": ", audio_nn_preds[1].confidence + "% ",
      audio_nn_preds[2].type + ": ", audio_nn_preds[2].confidence + "% ",
      audio_nn_preds[3].type + ": ", audio_nn_preds[3].confidence + "% ",
      audio_nn_preds[4].type + ": ", audio_nn_preds[4].confidence + "% ",
      //audio_nn_preds[5].type + ": ", audio_nn_preds[5].confidence + "% ",
      //audio_nn_preds[6].type + ": ", audio_nn_preds[6].confidence + "% ",
      //audio_nn_preds[7].type + ": ", audio_nn_preds[7].confidence + "% ",
      ]
    })

    // *************************************************************************** 
    // *************************** FINAL LIGHTGMB DATA ***************************
    // *************************************************************************** 

    // load, sort and set the data
    var data_lgbm = require('../../pipeline/pipeline_helper/final_tree_predictions.json')
    console.log("final lgbm predictions: " + data_lgbm)

    var lgbm_preds = [0, 0, 0, 0, 0, 0, 0, 0]

    for (var i = 0; i <= 7; i++) {
      lgbm_preds[i] = {
          type: AWS_RAVDESS_LIST[i], 
          confidence: data_lgbm[0][i].toFixed(2)
        }
    }

    lgbm_preds.sort((a, b) => (a.confidence) - (b.confidence)).reverse()

    console.log("sorted audio nn predictions: " + audio_nn_preds)

    this.setState({final_predictions: [
      "Final LightGBM Predictions: ",
      lgbm_preds[0].type + ": ", lgbm_preds[0].confidence + "% ",
      lgbm_preds[1].type + ": ", lgbm_preds[1].confidence + "% ",
      lgbm_preds[2].type + ": ", lgbm_preds[2].confidence + "% ",
      lgbm_preds[3].type + ": ", lgbm_preds[3].confidence + "% ",
      lgbm_preds[4].type + ": ", lgbm_preds[4].confidence + "% ",
      //lgbm_preds[5].type + ": ", lgbm_preds[5].confidence + "% ",
      //lgbm_preds[6].type + ": ", lgbm_preds[6].confidence + "% ",
      //lgbm_preds[7].type + ": ", lgbm_preds[7].confidence + "% ",
      ]
    })
  }

  render () {
    return (
      <div align='center'>
        <h2 className='title'>F.A.C.E. - Facial and Audio-based Classification of Emotion</h2>
        <h3 className='section-title'>UPLOAD PROCEDURE:</h3>
        <div className='main'>
        {/* UPLOAD SECTION */}
          <table className='table' align='center'>
            <tbody align='center'>
              <tr>
                <td className='table-cell'>
                  <Files
                    ref='files'
                    className='button-ui'
                    onChange={this.onFilesChange}
                    onError={this.onFilesError}
                    maxFiles={1}
                    maxFileSize={10000000}
                    minFileSize={0}
                    clickable
                  >
                    Choose File
                  </Files>
                </td>
                <td className='table-cell'>
                  <button className='button-ui'
                    style={{ height: '30px'}}
                    onClick={this.filesUpload}>Upload File</button>
                    {
                      this.state.files.length > 0
                      ? <div className='files-list'>
                        <div> {this.state.files.map((file) =>
                          <div className='files-list-item' key={file.id}>
                            <div className='files-list-item-content'>{file.name}</div>
                            <div
                              className='files-list-item-remove'
                              id={file.id}
                              onClick={this.filesRemoveOne.bind(this, file)} // eslint-disable-line
                            />
                          </div>
                        )}</div>
                      </div>
                      : null
                    }
                </td>
                <td className='table-cell'>
                  <button className='button-ui'
                    style={{height: '30px'}}
                    onClick={this.displayPipeline}>
                      Classify
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        {/* PIPELINE SECTION */}
        <h3 className='section-title'>PIPELINE:</h3>
        <div className='main'>
        <table className='table' align='center'>
        <tbody align='center'>
        {/* Pre-process image & audio files */}
        <tr>
        <td className='pipeline-table-cell'>
          <img border='none' className='pipeline-image' src={this.state.pre_processed_image[1]}/>
          {this.state.pre_processed_image[0]}
          <img className='pipeline-image' src={this.state.pre_processed_audio_image[1]}/>
          {this.state.pre_processed_audio_image[0]}
        </td>
        {/* AWS Predictions Table */}
        <td className='pipeline-table-cell'>
          <table>
            <tbody>
              <tr className='aws-pred-row'>
                <td className='aws-pred-cell' align='center'><b>{this.state.aws_results[0]}</b></td> 
              </tr>
              <tr className='aws-pred-row'>
                <td className='aws-pred-cell'>{this.state.aws_results[1]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[2]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[3]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[4]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[5]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[6]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[7]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[8]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[9]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[10]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[11]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[12]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[13]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[14]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.aws_results[15]}</td> 
                <td className='aws-pred-cell'>{this.state.aws_results[16]}</td>
              </tr>

            </tbody>
          </table>
        </td>
        {/* Audio Neural Network Predictions Table */}
        <td className='table-cell'>
        <table>
            <tbody>
              <tr className='aws-pred-row'>
                <td className='aws-pred-cell' align='center'><b>{this.state.audio_nn_results[0]}</b></td> 
              </tr>
              <tr className='aws-pred-row'>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[1]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[2]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[3]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[4]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[5]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[6]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[7]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[8]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[9]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[10]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[11]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[12]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[13]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[14]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.audio_nn_results[15]}</td> 
                <td className='aws-pred-cell'>{this.state.audio_nn_results[16]}</td>
              </tr>

            </tbody>
          </table>
        </td>
        {/* Final LGBM Predictions Table */}
        <td className='table-cell'>
        <table>
            <tbody>
              <tr className='aws-pred-row'>
                <td className='aws-pred-cell' align='center'><b>{this.state.final_predictions[0]}</b></td> 
              </tr>
              <tr className='aws-pred-row'>
                <td className='aws-pred-cell'>{this.state.final_predictions[1]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[2]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[3]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[4]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[5]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[6]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[7]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[8]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[9]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[10]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[11]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[12]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[13]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[14]}</td>
              </tr>
              <tr>
                <td className='aws-pred-cell'>{this.state.final_predictions[15]}</td> 
                <td className='aws-pred-cell'>{this.state.final_predictions[16]}</td>
              </tr>

            </tbody>
          </table>
        </td>       
        </tr>
        </tbody>
        </table>
      </div>
      <div className='final-classification'>
        {this.state.classifiction}
      </div>
      </div>
    )
  }
}

ReactDOM.render(<div><Application/></div>, document.getElementById('container'))
