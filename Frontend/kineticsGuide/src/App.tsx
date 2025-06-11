import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home/Home";
import Selection from "./pages/Selection/Selection";
import Exercise from "./pages/Exercise/Exercise";
import Login from "./pages/Login/Login";
import SignUp from "./pages/SignUp/SignUp";
import Developers from "./pages/Developers/Developers";
import Navbar from "./components/Navbar/Navbar";
import Footer from "./components/Footer/Footer";
import Record from "./pages/Record/Record";
import "./App.css"; // Import global styles

function App() {
  return (
    <div className="appContainer">
      <Navbar />
      <main className="mainContent">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/selection" element={<Selection />} />
          <Route path="/exercise/:id" element={<Exercise />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/developers" element={<Developers />} />
          <Route path="/record/:id" element={<Record />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;
