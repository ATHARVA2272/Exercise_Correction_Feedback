import React from "react";
import { Link } from "react-router-dom";
import styles from "./SignUp.module.css"; // Import modular CSS

const SignUp: React.FC = () => {
  return (
    <div className={styles.authContainer}>
      <div className={styles.authBox}>
        <h1 className={styles.authTitle}>Join KineticsGuide</h1>
        <p className={styles.authSubtitle}>Create an account and start tracking your workouts.</p>
        <form className={styles.authForm}>
          <input type="text" placeholder="Full Name" className={styles.authInput} />
          <input type="email" placeholder="Email" className={styles.authInput} />
          <input type="password" placeholder="Password" className={styles.authInput} />
          <button type="submit" className={styles.authButton}>Sign Up</button>
        </form>
        <p className={styles.authFooter}>
          Already have an account? <Link to="/login" className={styles.authLink}>Log in</Link>
        </p>
      </div>
    </div>
  );
};

export default SignUp;
