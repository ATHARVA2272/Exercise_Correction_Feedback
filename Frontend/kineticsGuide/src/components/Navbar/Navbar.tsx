import React from "react";
import { Link } from "react-router-dom";
import styles from "./Navbar.module.css";

const Navbar = () => {
  return (
    <nav className={styles.navbar}>
      <div className={styles.logo}>KineticsGuide</div>
      <ul className={styles.navLinks}>
        <li><Link to="/" className={styles.navItem}>Home</Link></li>
        <li><Link to="/selection" className={styles.navItem}>Exercises</Link></li>
        <li><Link to="/developers" className={styles.navItem}>Developers</Link></li>
      </ul>
      <div className={styles.authButtons}>
        <Link to="/login" className={`${styles.authButton} ${styles.loginButton}`}>Login</Link>
        <Link to="/signup" className={`${styles.authButton} ${styles.signupButton}`}>Sign Up</Link>
      </div>
    </nav>
  );
};

export default Navbar;
