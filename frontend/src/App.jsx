import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import NavBar from './components/NavBar';
import Home from './pages/Home';
import Graph from './pages/Graph';
import Datasets from './pages/Datasets';

import './App.css';

function App() {
  return (
    <Router>
      <NavBar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/graph" element={<Graph />} />
          <Route path="/datasets" element={<Datasets />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;