import { useState, useEffect } from 'react';

function Datasets() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/datasets')
      .then(res => res.json())
      .then(data => setDatasets(data))
      .catch(console.error);
  }, []);

  const loadDataset = async (filename) => {
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/datasets/${encodeURIComponent(filename)}`
      );

      if (!response.ok) throw new Error('Failed to load dataset');

      const data = await response.json();
      setSelectedDataset(data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Available Networks</h1>

      {datasets.map(filename => (
        <button
          key={filename}
          onClick={() => loadDataset(filename)}
        >
          {filename}
        </button>
      ))}

      {selectedDataset && (
        <div>
          <h2>{selectedDataset.description}</h2>

          <p>
            Total datapoints: {selectedDataset.datapoints.length}
          </p>

          <h3>Coordinates</h3>

          <ul>
            {selectedDataset.datapoints.map((point, index) => (
              <li key={index}>
                X: {point[0]}, Y: {point[1]}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Datasets;