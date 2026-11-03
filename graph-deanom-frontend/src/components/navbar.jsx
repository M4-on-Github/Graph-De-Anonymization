import { Link } from "react-router-dom";
import "../css/navbar.css";

function NavBar() { //TODO: Lots of CSS work
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">test, place icon here?</Link>
      </div>
      <div className="navbar-links">
        <Link to="/" className="nav-link">
          Home
        </Link>
        <Link to="/graph" className="nav-link">
          Graph two datasets 
        </Link>
        <Link to="/networks" className="nav-link">
          Look at avaliable networks
        </Link>
      </div>
    </nav>
  );
}

export default NavBar;
