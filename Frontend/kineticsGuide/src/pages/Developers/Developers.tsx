import React from 'react';
import styles from './Developers.module.css';

// Developer interface
interface Developer {
  name: string;
  department: string;
  photo: string;
}

const developers: Developer[] = [
  {
    name: 'Neeraj Ghatage',
    department: 'Computer Engineering',
    photo: '/src/assets/dhanashree.jpg',
  },
  {
    name: 'Atharva Jadhav',
    department: 'Computer Engineering',
    photo: '/src/assets/john_doe.jpg',
  },
  {
    name: 'Atharva Joshi',
    department: 'Computer Engineering',
    photo: '/src/assets/alice_smith.jpg',
  },
  {
    name: 'Dhanashree Kamble',
    department: 'Computer Engineering',
    photo: '/src/assets/alice_smith.jpg',
  },
];

const Developers = () => {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Our Developers</h1>
      <div className={styles.developersList}>
        {developers.map((developer, index) => (
          <div key={index} className={styles.developerCard}>
            <img
              src={developer.photo}
              alt={developer.name}
              className={styles.photo}
            />
            <h2 className={styles.name}>{developer.name}</h2>
            <p className={styles.department}>{developer.department}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Developers;
