import React from 'react';
import { Container, Typography, Grid, Button, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import styles from './Home.module.css';
import { useNavigate } from "react-router-dom";

const Home = () => {
  const navigate = useNavigate();
  return (
    <Container maxWidth="lg">
      {/* Hero Section */}
      <section className={styles.heroSection}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <Typography variant="h3" fontWeight="bold">
            Kinetics<span style={{ color: '#ffcc00' }}>Guide</span>: Your AI-Powered Workout Companion
          </Typography>
          <Typography variant="h6" sx={{ mt: 2, opacity: 0.8 }}>
            Enhance Your Workouts with Real-Time Posture Analysis & Corrective Feedback
          </Typography>
          <Button
            variant="contained"
            sx={{
              mt: 4,
              bgcolor: '#ffcc00',
              color: '#222',
              fontWeight: 'bold',
              '&:hover': { bgcolor: '#ffb300' },
            }}
            onClick={() => navigate("/selection")}
          >
            Get Started
          </Button>
        </motion.div>
      </section>

      {/* Features Section */}
      <Typography variant="h4" textAlign="center" fontWeight="bold" sx={{ mt: 6 }}>
        Why Choose <span style={{ color: '#ff4b2b' }}>KineticsGuide?</span>
      </Typography>
      <Grid container spacing={4} sx={{ mt: 3 }}>
        {[
          { title: 'Real-Time Posture Detection', desc: 'AI-powered tracking for various exercises.' },
          { title: 'Instant Corrective Feedback', desc: 'Receive guidance through visual and audio cues.' },
          { title: 'Injury Prevention', desc: 'Maintain proper form and avoid injuries.' },
          { title: 'Personalized Training', desc: 'Custom feedback tailored to your fitness goals.' },
          { title: 'Seamless Integration', desc: 'Works with standard webcams and mobile cameras.' },
        ].map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <motion.div whileHover={{ scale: 1.05 }} transition={{ duration: 0.3 }}>
              <Paper className={styles.featureCard} elevation={3}>
                <Typography variant="h6" fontWeight="bold" color="#ff4b2b">
                  {feature.title}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1, opacity: 0.7 }}>
                  {feature.desc}
                </Typography>
              </Paper>
            </motion.div>
          </Grid>
        ))}
      </Grid>

      {/* How It Works */}
      <Typography variant="h4" textAlign="center" fontWeight="bold" sx={{ mt: 8, mb: 4 }}>
        How It Works
      </Typography>
      <Grid container spacing={4} justifyContent="center" sx={{ mt: 3 }}>
        {[
          { step: '1️⃣ Capture', desc: 'Record using a webcam or phone.' },
          { step: '2️⃣ Analyze', desc: 'AI assesses posture, joint angles.' },
          { step: '3️⃣ Correct', desc: 'Get instant feedback for better form.' },
          { step: '4️⃣ Improve', desc: 'Track progress & refine technique.' },
        ].map((step, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <motion.div whileHover={{ scale: 1.1 }} transition={{ duration: 0.3 }}>
              <Paper className={styles.featureCard} elevation={2}>
                <Typography variant="h6" fontWeight="bold" color="#ff4b2b">
                  {step.step}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1, opacity: 0.7 }}>
                  {step.desc}
                </Typography>
              </Paper>
            </motion.div>
          </Grid>
        ))}
      </Grid>

      {/* Call to Action Section */}
      <section className={styles.ctaSection} sx={{ mt: 4 }}>
  <Typography variant="h4" fontWeight="bold">
    Unlock Your Full Potential with KineticsGuide
  </Typography>
  <Typography variant="h6" sx={{ mt: 2, opacity: 0.8 }}>
    Join the future of smart fitness and revolutionize your workouts with AI-powered precision.
  </Typography>
  <motion.div whileHover={{ scale: 1.1 }}>
    <Button
      variant="contained"
      sx={{
        mt: 4,
        bgcolor: 'black',
        color: 'white',
        fontWeight: 'bold',
        '&:hover': { bgcolor: '#222' },
      }}
      onClick={() => navigate("/selection")}
    >
      Start Training Now
    </Button>
    
  </motion.div>
</section>
    </Container>
  );
};

export default Home;
