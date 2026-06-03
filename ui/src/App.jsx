import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [game, setGame] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [mirrorLine, setMirrorLine] = useState("Mirror, mirror... shall we begin?");

  async function startGame() {
    setLoading(true);
    setMirrorLine("The mirror awakens...");

    const response = await fetch(`${API_URL}/game/start`, {
      method: "POST",
    });

    const data = await response.json();
    setGame(data);
    setMirrorLine("I see a shadow forming...");
    setLoading(false);
  }

  async function sendAnswer(answer) {
    if (!game?.game_id) return;

    setLoading(true);
    setMirrorLine("The mirror is reading your answer...");

    const response = await fetch(`${API_URL}/game/answer`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        game_id: game.game_id,
        answer: answer,
      }),
    });

    const data = await response.json();
    setGame(data);
    setMirrorLine("The vision grows clearer...");
    setLoading(false);
  }
  
  async function sendConfirm(correct) {
    if (!game?.game_id) return;
  
    const res = await fetch(`${API_URL}/game/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        game_id: game.game_id,
        correct: correct,
      }),
    });
  
    const data = await res.json();
  
    setGame(data);
  }

  async function confirmGuess(correct) {
    if (!game?.game_id) return;

    setLoading(true);

    const response = await fetch(`${API_URL}/game/confirm`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        game_id: game.game_id,
        correct: correct,
      }),
    });

    const data = await response.json();
    setGame(data);

    if (correct) {
      setMirrorLine("Of course I knew it. I am the Magic Mirror... and now, apparently, AI.");
    } else {
      setMirrorLine("A crack in the prophecy... the vision was clouded.");
    }

    setLoading(false);
  }

  const status = game?.status || "idle";
  const question = game?.question;
  const guess = game?.guess || game?.result;
  const history = game?.history || [];

  return (
    <main className="app">
      <section className="mirror-stage">
        <h1 className="title">Disney Villain Magic Mirror</h1>
  
        <div className="game-layout">
          <div
            className={`mirror-shell ${
              status === "guessing" || status === "done" ? "guessing" : ""
            }`}
          >
            <div className="mirror-glass">
              <div className="mist"></div>
  
              <div className="mirror-content">
                {status === "idle" && (
                  <>
                    <p className="mirror-line">Mirror, mirror... shall we begin?</p>
                    <button className="primary-btn" onClick={startGame}>
                      Start the prophecy
                    </button>
                  </>
                )}
  
                {status === "asking" && question && (
                  <>
                    <p className="mirror-line">I see a shadow forming...</p>
                    <p className="mirror-line">Answer carefully, mortal.</p>
                  </>
                )}
  
                {status === "guessing" && guess && (
                  <>
                    <p className="mirror-line">The vision is clear...</p>
                    <h2 className="guess-text">
                      Are you thinking of <span>{guess.villain}</span>?
                    </h2>
                  </>
                )}
  
                {status === "done" && guess && (
                  <>
                    <p className="mirror-line">
                      {guess.correct_guess
                        ? "Of course I knew it. I am the Magic Mirror... and now, apparently, AI."
                        : "A crack in the prophecy... the vision was clouded."}
                    </p>
  
                    <h2 className="guess-text">
                      Final Guess: <span>{guess.guessed_villain || "Unknown"}</span>
                    </h2>
  
                    <button className="primary-btn" onClick={startGame}>
                      Play Again
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
  
          {status === "asking" && question && (
            <div className="outside-question-card">
              <p className="card-label">The mirror asks...</p>
              <h2 className="question-text">{question.text}</h2>
  
              <div className="outside-answer-panel">
                <button onClick={() => sendAnswer("yes")}>Yes</button>
                <button onClick={() => sendAnswer("prob_yes")}>Probably Yes</button>
                <button onClick={() => sendAnswer("idk")}>I Don’t Know</button>
                <button onClick={() => sendAnswer("prob_no")}>Probably No</button>
                <button onClick={() => sendAnswer("no")}>No</button>
              </div>
            </div>
          )}
  
          {status === "guessing" && guess && (
            <div className="outside-question-card">
              <p className="card-label">Confirm the prophecy...</p>
              <div className="outside-answer-panel">
                <button onClick={() => sendConfirm(true)}>Yes, correct</button>
                <button onClick={() => sendConfirm(false)}>No, try again</button>
              </div>
            </div>
          )}
        </div>
  
        {history.length > 0 && (
          <details className="history-section">
            <summary className="history-toggle">Prophecy Log</summary>
  
            <div className="history-log">
              {history.map((item, index) => (
                <div className="history-item" key={index}>
                  <strong>{item.question || item.type}</strong>
                  {item.answer && <p>Answer: {item.answer}</p>}
                  {item.remaining_candidates !== undefined && (
                    <p>Remaining visions: {item.remaining_candidates}</p>
                  )}
                </div>
              ))}
            </div>
          </details>
        )}
      </section>
    </main>
  );
}

export default App;