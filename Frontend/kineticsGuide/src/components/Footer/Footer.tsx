import React from "react";
import styles from "./Footer.module.css"; // Import the module CSS

const Footer: React.FC = () => {
  return (
    <footer className={styles.footer}>
      <p>© {new Date().getFullYear()} Kinetics Guide. All rights reserved.</p>
    </footer>
  );
};

export default Footer;
