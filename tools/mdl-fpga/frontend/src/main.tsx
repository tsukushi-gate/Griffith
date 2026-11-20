import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './index.css';
import './styles.css';
import App from './App';
import { ErrorBoundary, GlobalErrorOverlay } from './ErrorBoundary';
import { initTheme } from './store/themeStore';

initTheme();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <GlobalErrorOverlay />
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
