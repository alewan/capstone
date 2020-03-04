import axios from 'axios'
import Blob from 'blob'
import FormData from 'form-data'
import React from 'react'
import ReactDOM from 'react-dom'
import Files from './file'
// import Classification from './class'

class Application extends React.Component {
  constructor (props) {
    super(props)
    this.state = {
      files: [], 
      classifiction: ""
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

  classifyEmotion = (classifiction) => {
    console.log("um hello")
    console.log(this.state.classifiction)

    axios.post('/classify')
    //.then(response => window.alert(response.data.result))
    .then(response => this.setState({classifiction: "Emotion Prediction: " + response.data.result}))
    //.then(this.setState({classifiction: response.data.result}))
    .then(console.log(this.state.classifiction))
  }

  render () {
    return (
      <div align='center'>
        <h2 className='title'> F.A.C.E. - Facial and Audio-based Classification of Emotion </h2>
        <div className='main'>
        <table className='table' align='center'>
        <tbody align='center'>
        <tr>
          <td className='table-cell'> Step 1. Select File: </td>
          <td className='table-cell'>  Step 2. Upload File: </td>
          <td className='table-cell'> Step 3. Classify Emotion: </td>
        </tr>
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
          style={{ height: '30px'}}
          onClick={this.classifyEmotion}>
            Classify
          </button>
        </td>
        </tr>
         <tr> {/* third row */}
        <td className='table-cell'> </td>
        <td className='table-cell' align='center'> {this.state.classifiction} </td>
        <td className='table-cell'> </td>
        </tr>
        </tbody>
        </table>
        </div>
      </div>
    )
  }
}

ReactDOM.render(<div><Application /></div>, document.getElementById('container'))
