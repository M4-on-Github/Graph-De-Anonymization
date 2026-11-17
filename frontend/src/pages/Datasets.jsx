import { useEffect, useState } from 'react';

function Datasets() {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    fetch('/datasets')
      .then(res => res.json())
      .then(data => setFiles(data))
      .catch(console.error);
  }, []);

  return (
    <div>
      <h1>Available Networks</h1>
      <ul>
        {files.map(fileName => (
          <li key={fileName}>{fileName}</li>
        ))}
      </ul>
    </div>
  );
}

export default Datasets;