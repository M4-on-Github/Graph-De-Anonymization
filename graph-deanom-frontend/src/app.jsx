import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import NavBar from './components/NavBar';
import './App.css'

function App() {

  return (
    <div>
    <Router>
      <NavBar/>
      <Routes>
        <Route path="/"/>
        <Route path="/graphs"/>
      </Routes>
    </Router>
    </div>
  );
}

export default App