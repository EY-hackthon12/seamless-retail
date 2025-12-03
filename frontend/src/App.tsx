import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import MobileChat from './components/MobileChat';
import KioskDashboard from './components/KioskDashboard';

function App() {
  return (
    <Router>
      <div className="min-h-screen">
        <nav className="bg-primary text-white p-4 shadow-md">
          <div className="container mx-auto flex justify-between items-center">
            <h1 className="text-xl font-bold tracking-tight">UrbanVogue AI</h1>
            <div className="space-x-4">
              <Link to="/mobile" className="hover:text-accent transition-colors">Mobile App</Link>
              <Link to="/kiosk" className="hover:text-accent transition-colors">In-Store Kiosk</Link>
            </div>
          </div>
        </nav>

        <main className="container mx-auto p-4">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/mobile" element={<MobileChat />} />
            <Route path="/kiosk" element={<KioskDashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

function Home() {
  return (
    <div className="flex flex-col items-center justify-center h-[80vh] space-y-8">
      <h2 className="text-4xl font-bold text-center">Experience Seamless Retail</h2>
      <p className="text-xl text-gray-600 max-w-2xl text-center">
        Start your journey on the mobile app, and continue seamlessly at our in-store kiosk.
        Our Cognitive Retail Brain remembers your context.
      </p>
      <div className="flex space-x-6">
        <Link to="/mobile" className="px-8 py-4 bg-primary text-white rounded-lg shadow-lg hover:bg-secondary transition-all transform hover:scale-105">
          Launch Mobile App
        </Link>
        <Link to="/kiosk" className="px-8 py-4 bg-white text-primary border-2 border-primary rounded-lg shadow-lg hover:bg-gray-50 transition-all transform hover:scale-105">
          Launch Kiosk
        </Link>
      </div>
    </div>
  );
}

export default App;
