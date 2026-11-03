import { Link } from "react-router-dom";
import "../css/navbar.css";

function NavBar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">Network Visualizer</Link>
      </div>

      <div className="navbar-links">
        <Link to="/" className="nav-link">
          Home
        </Link>

        <Link to="/graph" className="nav-link">
          Graph Datasets
        </Link>

        <Link to="/datasets" className="nav-link">
          Available Networks
        </Link>
      </div>
    </nav>
  );
}

export default NavBar;