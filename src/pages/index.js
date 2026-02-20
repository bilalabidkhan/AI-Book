import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';
import { useState, useEffect } from "react";

function HomepageHeader() {
  const [fontSize, setFontSize] = useState("3.7rem");

  useEffect(() => {
    function handleResize() {
      if (window.innerWidth <= 480) setFontSize("1.8rem");
      else if (window.innerWidth <= 996) setFontSize("2.2rem");
      else setFontSize("3.7rem");
    }
    window.addEventListener("resize", handleResize);
    handleResize();
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)} style={{background: "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"}}>
      
       <div className="container">
        <h1 className="hero__title" style={{color: "white", fontSize: fontSize }}>Physical AI & Humanoid Robotics Textbook</h1>

        <p style={{ maxWidth: '790px', margin: '2rem auto', fontSize: '1.2rem', color: "white" }}>        
          A complete guide to humanoid robotics covering 
          ROS 2 integration, Digital Twins, AI-Robot Brains, 
          and Vision-Language-Action systems — from simulation 
          to real-world intelligence.
        </p>

        <div className={styles.bounce}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/">
            📘 Start Reading
          </Link>
        </div>
      </div>
    </header>
  );
}

function ModulesSection() {
  return (
    <section className={styles.modulesSection}>
      <div className="container">
       <h2 className={styles.sectionTitle}>📚 Explore All Modules</h2>
        <div className={styles.moduleGrid}>

          <div className={styles.moduleCard}>
            <Link to='/docs/modules/ros2-humanoid-integration/'>
             <h3>🦾 Module 1: ROS 2 Humanoid Integration</h3>
            </Link>
            <ul>
              <li>ROS 2 Fundamentals</li>
              <li>Python Agents with ROS 2</li>
              <li>Humanoid Modeling with URDF</li>
            </ul>
          </div>

          <div className={styles.moduleCard}>
            <Link to="/docs/module-2-digital-twin">
             <h3>🌐 Module 2: Digital Twin</h3>
            </Link>
            <ul>
              <li>Gazebo Physics Simulation</li>
              <li>Sensor Simulation (LiDAR, IMU)</li>
              <li>Unity High-Fidelity Environments</li>
            </ul>
          </div>

          <div className={styles.moduleCard}>
            <Link to="/docs/module-3-ai-robot-brain">
             <h3>🧠 Module 3: AI-Robot Brain</h3>
            </Link>
            <ul>
              <li>NVIDIA Isaac Sim</li>
              <li>Isaac ROS & VSLAM</li>
              <li>Nav2 Navigation</li>
            </ul>
          </div>

          
          <div className={styles.moduleCard}>
            <Link to="/docs/module-4-vla">
             <h3>🗣️ Module 4: Vision-Language-Action</h3>
            </Link>
            <ul>
              <li>Voice-to-Action</li>
              <li>Cognitive Planning with LLMs</li>
              <li>Vision-Guided Manipulation</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="A comprehensive guide to AI and Robotics">
      <HomepageHeader />
      <ModulesSection />
      <main>
        {/* Additional content can be added here */}
      </main>
    </Layout>
  );
}