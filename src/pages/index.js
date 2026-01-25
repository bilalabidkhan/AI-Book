import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      
       <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>

        <p className="hero__subtitle">
          🤖 Physical AI & Humanoid Robotics
        </p>

        <p style={{ maxWidth: '820px', margin: '1rem auto', fontSize: '1.1rem' }}>
          A structured, university-level textbook on building intelligent humanoid robots —
          from <strong>ROS 2 fundamentals</strong> to <strong> Digital Twins</strong>,
          <strong> AI-Robot Brains</strong>, and <strong>Vision-Language-Action systems</strong>.
        </p>

        <p style={{ opacity: 0.9, marginBottom: '2rem' }}>
          Designed for students, researchers, and developers working at the
          intersection of AI and Robotics.
        </p>

        <div className="hero__buttons">
          <Link
            className="button button--primary button--lg"
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
       <h2 className={styles.sectionTitle}>📚 Course Modules</h2>
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