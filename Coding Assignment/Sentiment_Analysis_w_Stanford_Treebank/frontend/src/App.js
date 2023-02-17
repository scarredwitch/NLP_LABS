import './App.css';
import React from 'react';
import { useState } from 'react';
import axios from 'axios';

function App() {
  const [inputText, setInputText] = useState("");
  const [image, setImage] = useState("");
  const [topBadWords, setTopBadWords] = useState([])
  const [topGoodWords, setTopGoodWords] = useState([])

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    e.stopPropagation();

    axios.post(`http://localhost:8000/process_data?text=${inputText}`).then((res) => {
      console.log(res.data);
      setImage(res.data.images);
      setTopBadWords(res.data.Topbad);
      setTopGoodWords(res.data.topgood);
    }).catch((err)=>{
      console.log(err)
    });
  }
  return (
   <div className='container-fluid'>
     <form className='pt-3 mb-3 row g-3' onSubmit={handleSubmit}>
      <div className='col-auto'>
        <label className='form-label'>
          Enter text:
        </label>
      </div>
      <div  className='col-auto'>
      <input type="text" className='px-2 form-control' value={inputText} onChange={handleInputChange} />
      </div>
      <div className='col-auto'>
        <button type="submit" className='btn btn-primary mb-3'>Submit</button>
      </div>
      <div className='pt-5'>
      <table className="table table-primary">
        <thead>
          <tr>
            <th>
              Top bad words
            </th>
            <th>Top good words</th>
          </tr>
        </thead>
        <tbody>
        {topBadWords.map((word,index)=>(
          <tr>
            <td>{word}</td>
            <td>{topGoodWords[index]}</td>
          </tr>
        ))}
        </tbody>
       
      </table>
      {image ? <img className='img-fluid rounded mx-auto d-block' src={`data:image/png;base64,${image}`}/>: ''}
      </div>
    </form>
   </div>
  );
}

export default App;