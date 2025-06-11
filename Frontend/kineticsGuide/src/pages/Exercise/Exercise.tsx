import React from "react";
import { useParams, useNavigate } from "react-router-dom";
import { exercises } from "../../data/exercises";
import styles from "./Exercise.module.css"; // Ensure lowercase "module.css"
import { FaDumbbell, FaCheckCircle, FaExclamationTriangle } from "react-icons/fa";

const Exercise = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const exercise = exercises[id as keyof typeof exercises];
  console.log(exercise);
  if (!exercise) {
    return (
      <div className={styles.notFound}>
        <h1>Exercise Not Found</h1>
      </div>
    );
  }

  const handleStartRecording = () => {
    navigate(`/record/${id}`);
  };

  return (
    <div className={styles.container}>
      {/* Title & Description with Button */}
<h1 className={styles.title}>{exercise.name}  </h1>
<div className={styles.descriptionContainer}>
  <p className={styles.description}>{exercise.description} </p>
  
</div>


     {/* Image & Video Side-by-Side */}
<div className={styles.mediaContainer}>
  {/* Image Section */}
  <div className={styles.imageContainer}>
    <img src={exercise.image} alt={exercise.name} className={styles.image} />
  </div>

  {/* Video Section */}
  <div className={styles.videoContainer}>
    {exercise.video.includes("youtube.com") ? (
      <iframe
        className={styles.video}
        src={exercise.video.replace("watch?v=", "embed/")}
        title={exercise.name}
        allowFullScreen
      ></iframe>
    ) : (
      <video controls className={styles.video}>
        <source src={exercise.video} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    )}
  </div>
</div>


      {/* Information Grid */}
      <div className={styles.grid}>
        {/* Muscle Groups */}
        <div className={styles.card}>
          <h2>Targeted Muscles</h2>
          <ul>
            {exercise.muscleGroups.map((muscle, index) => (
              <li key={index}>
                <FaDumbbell className={styles.icon} /> {muscle}
              </li>
            ))}
          </ul>
        </div>

        {/* Equipment */}
        <div className={styles.card}>
          <h2>Required Equipment</h2>
          <ul>
            {exercise.equipment.map((item, index) => (
              <li key={index}>
                <FaCheckCircle className={styles.icon} /> {item}
              </li>
            ))}
          </ul>
        </div>

        {/* How to Perform */}
        <div className={`${styles.card} ${styles.fullWidth}`}>
          <h2>How to Perform</h2>
          <ul>
            {exercise.howToPerform.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ul>
        </div>

        {/* Common Mistakes */}
        <div className={styles.card}>
          <h2>Common Mistakes</h2>
          <ul>
            {exercise.commonMistakes.map((mistake, index) => (
              <li key={index}>
                <FaExclamationTriangle className={styles.warningIcon} /> {mistake}
              </li>
            ))}
          </ul>
        </div>

        {/* Benefits */}
        <div className={styles.card}>
          <h2>Benefits</h2>
          <ul>
            {exercise.benefits.map((benefit, index) => (
              <li key={index}>
                <FaCheckCircle className={styles.benefitIcon} /> {benefit}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Start Recording Button */}
      <button onClick={handleStartRecording} className={styles.button}>
        Start Recording
      </button>
    </div>
  );
};

export default Exercise;
