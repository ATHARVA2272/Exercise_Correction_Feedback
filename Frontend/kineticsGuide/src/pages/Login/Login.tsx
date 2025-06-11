import React from "react";
import { Link } from "react-router-dom";
import styles from "./Login.module.css"; // Importing modular CSS

const Login: React.FC = () => {
  return (
    <div className={styles.authContainer}>
      <div className={styles.authBox}>
        <h1 className={styles.authTitle}>Welcome Back!</h1>
        <p className={styles.authSubtitle}>Log in to continue your fitness journey.</p>
        <form className={styles.authForm}>
          <input type="email" placeholder="Email" className={styles.authInput} />
          <input type="password" placeholder="Password" className={styles.authInput} />
          <button type="submit" className={styles.authButton}>Log In</button>
        </form>
        <p className={styles.authFooter}>
          New here? <Link to="/signup" className={styles.authLink}>Sign up</Link>
        </p>
      </div>
    </div>
  );
};

export default Login;
