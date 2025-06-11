export interface Exercise {
  name: string;
  description: string;
  image: string;
  video: string;
  muscleGroups: string[];
  equipment: string[];
  howToPerform: string[];
  commonMistakes: string[];
  benefits: string[];
}

export const exercises: Record<string, Exercise> = {
  lunges: {
    name: "Lunges",
    description: "Lunges improve lower-body strength ",
    image: "/src/assets/lunges.jpg",
    video: "https://www.youtube.com/embed/wrwwXE_x-pQ",
    muscleGroups: ["Quadriceps", "Glutes", "Hamstrings", "Core"],
    equipment: ["Bodyweight", "Dumbbells (optional)"],
    howToPerform: [
      "Stand tall, step one foot forward, and lower your hips until both knees are bent at 90 degrees.",
      "Keep your back straight, core engaged, and front knee aligned over the ankle.",
      "Push back up to the starting position and repeat on the other leg."
    ],
    commonMistakes: [
      "Knees extending beyond toes",
      "Leaning forward",
      "Lack of control during movement"
    ],
    benefits: [
      "Improves lower body strength",
      "Enhances balance",
      "Increases flexibility"
    ]
  },
  bicepCurl: {
    name: "Bicep Curl",
    description: "Bicep curls help strengthen your arms.",
    image: "/src/assets/bicepCurl.jpeg",
    video: "https://www.youtube.com/embed/XE_pHwbst04",
    muscleGroups: ["Biceps"],
    equipment: ["Dumbbells", "Resistance Bands (optional)"],
    howToPerform: [
      "Hold dumbbells with palms facing forward and elbows close to your torso.",
      "Curl the weights toward your shoulders, keeping your upper arms stationary.",
      "Slowly lower the weights back to the starting position."
    ],
    commonMistakes: [
      "Swinging arms",
      "Using momentum",
      "Improper wrist alignment"
    ],
    benefits: [
      "Increases arm strength",
      "Enhances muscle definition",
      "Improves grip strength"
    ]
  },
  squat: {
    name: "Squat",
    description: "Squats are a great way to build leg strength.",
    image: "/src/assets/squat.jpeg",
    video: "https://www.youtube.com/embed/YaXPRqUwItQ",
    muscleGroups: ["Quadriceps", "Glutes", "Hamstrings", "Core"],
    equipment: ["Bodyweight", "Barbell/Dumbbells (optional)"],
    howToPerform: [
      "Stand with feet shoulder-width apart, toes slightly pointed out.",
      "Lower your hips as if sitting in a chair, keeping your back straight and chest up.",
      "Push through your heels to return to the standing position."
    ],
    commonMistakes: [
      "Knees caving in",
      "Heels lifting off the ground",
      "Rounding the back"
    ],
    benefits: [
      "Builds lower body strength",
      "Enhances mobility",
      "Improves posture"
    ]
  },
  plank: {
    name: "Plank",
    description: "Planks are great for core strength.",
    image: "/src/assets/plank.jpeg",
    video: "https://www.youtube.com/embed/6jXj2Ki2T80",
    muscleGroups: ["Core", "Shoulders", "Glutes", "Back"],
    equipment: ["None"],
    howToPerform: [
      "Lie face down and lift your body onto your forearms and toes, keeping your body in a straight line.",
      "Engage your core, keep your back neutral, and hold the position.",
      "Avoid sagging hips or arching your back."
    ],
    commonMistakes: [
      "Dropping hips",
      "Excessive arching",
      "Holding breath"
    ],
    benefits: [
      "Strengthens core stability",
      "Improves posture",
      "Enhances endurance"
    ]
  }
};
