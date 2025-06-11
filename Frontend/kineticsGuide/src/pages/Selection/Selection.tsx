import React from "react";
import { Link } from "react-router-dom";
import styles from "./Selection.module.css";

// Import images correctly
import lungesImg from "../../assets/lunges.jpg";
import bicepCurlImg from "../../assets/bicepCurl.jpeg";
import squatImg from "../../assets/squat.jpeg";
import plankImg from "../../assets/plank.jpeg";

const exercises = [
  { name: "Lunges", image: lungesImg, path: "/exercise/lunges" },
  { name: "Bicep Curl", image: bicepCurlImg, path: "/exercise/bicepCurl"},
  { name: "Squat", image: squatImg, path: "/exercise/squat" },
  { name: "Plank", image: plankImg, path: "/exercise/plank" },
];

const Selection: React.FC = () => {
  return (
    <div className={styles.selectionContainer}>
      <h1 className={styles.title}>Select Your Exercise</h1>
      <div className={styles.exerciseGrid}>
        {exercises.map((exercise) => (
          <Link to={exercise.path} key={exercise.name} className={styles.exerciseCard}>
            <img src={exercise.image} alt={exercise.name} className={styles.exerciseImage} />
            <h2 className={styles.exerciseName}>{exercise.name}</h2>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Selection;
