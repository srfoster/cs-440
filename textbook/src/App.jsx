import React, { useState, useEffect } from 'react';
import { HashRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import {
  ExamDashboard,
  ExamInterface,
  TextbookPage,
  compiledContentService,
  loadAllQuestions
} from '@srfoster/textbook-lib';
import { compiledFiles, stats } from './compiled';

// Define CS-440 concept map paths
const CONCEPT_MAP_PATHS = [
  'content/chapter-01/concept-map.yml',
  'content/chapter-02/concept-map.yml',
  'content/chapter-03/concept-map.yml'
];

function AppContent() {
  const location = useLocation();
  const [questions, setQuestions] = useState([]);
  const [currentExam, setCurrentExam] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Initialize the compiled content service with our compiled content and base path
    compiledContentService.initialize({ 
      compiledFiles, 
      stats,
      basePath: import.meta.env.BASE_URL || '/'
    });
    
    const fetchQuestions = async () => {
      try {
        const allQuestions = await loadAllQuestions(CONCEPT_MAP_PATHS);
        setQuestions(allQuestions);
      } catch (error) {
        console.error('Failed to load questions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchQuestions();
  }, []);

  if (loading) {
    return (
      <div className="app loading">
        <h2>Loading CS440 AI Textbook...</h2>
      </div>
    );
  }

  const handleStartExam = (examConfig) => {
    setCurrentExam(examConfig);
  };

  const handleEndExam = () => {
    setCurrentExam(null);
  };

  const isExamRoute = location.pathname === '/exam' || location.pathname === '/#/exam';

  return (
    <div className="app">
      <header>
        <div className="header-content">
          <div className="header-title">
            <h1>CS440: Artificial Intelligence</h1>
            <p>Exploring intelligent agents, search algorithms, and machine learning</p>
          </div>
          <nav className="main-nav">
            <Link to="/textbook" className={`nav-link ${location.pathname.startsWith('/textbook') ? 'active' : ''}`}>
              Textbook
            </Link>
            <Link to="/exam" className={`nav-link ${isExamRoute ? 'active' : ''}`}>
              Practice Exam
            </Link>
          </nav>
        </div>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/textbook" replace />} />
          <Route path="/textbook/*" element={<TextbookPage />} />
          <Route 
            path="/exam" 
            element={
              currentExam ? (
                <ExamInterface 
                  questions={currentExam.questions} 
                  settings={currentExam.settings}
                  onEndExam={handleEndExam}
                />
              ) : (
                <ExamDashboard 
                  questions={questions} 
                  onStartExam={handleStartExam}
                  courseTitle="CS440: Artificial Intelligence"
                />
              )
            } 
          />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
