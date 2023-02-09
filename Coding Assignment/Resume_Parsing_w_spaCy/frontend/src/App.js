import './App.css';
import React from 'react';
import { useState } from 'react';
import axios from 'axios';

function App() {
  // drag state
  const [dragActive, setDragActive] = React.useState(false);
  // ref
  const inputRef = React.useRef(null);
  const [educationList, setEducationList] = React.useState([])
  const [skillsList, setSkillsList] = React.useState([])
  const [showDownloadButton, setShowDownloadButton] = React.useState(false);
  const [resumes, setResumes] = useState([]);
  // handle drag events
  const handleDrag = function (e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // triggers when file is dropped
  const handleDrop = function (e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length >= 1) {
      console.log(e.dataTransfer.files);
      // handleFiles(e.dataTransfer.files);
      // setResumes((resumes) => [...resumes, ...e.dataTransfer.files]);
      setResumes(e.dataTransfer.files[0])
    }
  };

  // triggers when file is selected with click
  const handleChange = function (e) {
    e.preventDefault();
    console.log(e.target.files);
    debugger;
    if (e.target.files && e.target.files.length >= 1) {
      // setResumes((resumes) => [...resumes, ...e.target.files]);
      setResumes(e.target.files[0])
    }
  };

  // triggers the input when the button is clicked
  const onButtonClick = () => {
    inputRef.current.click();
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    // const totalResumes = resumes.length;
    formData.append('pdf', resumes);
    console.log(resumes.type)

    //setShowDownloadButton(true);
    // formData.append('file', e.target.file[0]);
    axios.post('http://localhost:8000/pdf_to_dataframe', formData,{
      headers:{
        'Content-Type':resumes.type
      }
    }).then((res) => {
      console.log(res.data)
      setEducationList(res.data.education_status);
      const skills = res.data.skills.filter((item, i, skills) => skills.indexOf(item) === i)
      setSkillsList(skills)
    }).catch((err)=>{
      console.log(err)
    });
  };

  return (
    <div className="page">
      <h1>Resume Extractor</h1>
      <h4>Upload your resume to extract your Education and Skills automatically.</h4>
      <form id="form-file-upload" onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
        <input
          ref={inputRef}
          type="file"
          id="input-file-upload"
          accept=".pdf"
          onChange={handleChange}
        />
        <label
          id="label-file-upload"
          htmlFor="input-file-upload"
          className={dragActive ? 'drag-active' : ''}
        >
          <div>
            <p>Drag and drop your resume here or</p>
            <button className="upload-button" onClick={onButtonClick}>
              Upload a Resume
            </button>
          </div>
        </label>
        {dragActive && (
          <div
            id="drag-file-element"
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          ></div>
        )}
      </form>
      {showDownloadButton ? (
          <button type="button" id="download-resume">
            Download
          </button>
        ) : (
          <button type="button" id="submit-resume" onClick={onSubmit}>
            Submit
          </button>
        )}
      <div className='display-div'>
        <div className='grid-item education-div'>
          <h2>Education</h2>
        <ol className='education-list'>

          {educationList && educationList.map(education =>
                
                <li className={`${education} list-item`}>{education}</li>
            )}
                        </ol>

        </div>
        <div className='grid-item skill-div'>
          <h2>Skills</h2>
        <ol className='skill-list'>
          {skillsList && skillsList.map(skill =>
              
                <li className={`${skill} list-item`}>{skill}</li>
            )}
                        </ol>
        </div>
      </div>
      
    </div>
  );
}

export default App;