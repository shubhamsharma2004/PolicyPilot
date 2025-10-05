import { useEffect, useMemo, useRef, useState } from "react";
import { ask } from "./api";
import "./index.css";

export default function App() {
  // ————— THEME
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);
  const toggleTheme = () => setTheme(t => (t === "dark" ? "light" : "dark"));

  // ————— APP STATE
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const ctrlRef = useRef(null);

  const canAsk = q.trim().length > 0 && !loading;

  const onAsk = async () => {
    if (!canAsk) return;
    setLoading(true);
    setErr("");
    ctrlRef.current?.abort?.();
    ctrlRef.current = new AbortController();
    try {
      const data = await ask(q.trim(), ctrlRef.current.signal);
      setRes(data);
    } catch {
      setErr("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const onClear = () => {
    setQ("");
    setRes(null);
    setErr("");
  };

  const copyAnswer = async () => {
    if (!res?.answer) return;
    await navigator.clipboard.writeText(res.answer);
    toast("Answer copied");
  };

  const title = useMemo(() => "PolicyPilot", []);

  return (
    <div className="page">
      {/* animated bg */}
      <div className="bg">
        <div className="blob a" />
        <div className="blob b" />
        <div className="scanline" />
      </div>

      <header className="topbar">
        <Logo />
        <div className="brand">
          <h1>{title}</h1>
          <span className="sub">Company Policy Assistant</span>
        </div>

        <div className="top-actions">
          <button
            className="btn icon"
            aria-label="Toggle theme"
            title={theme === "dark" ? "Switch to light" : "Switch to dark"}
            onClick={toggleTheme}
          >
            {theme === "dark" ? "🌞" : "🌙"}
          </button>
        </div>
      </header>

      <main className="container">
        {/* card: question */}
        <section className="card query">
          <label className="label">Ask a question</label>
          <div className="inputRow">
            <input
              className="input"
              value={q}
              onChange={e => setQ(e.target.value)}
              placeholder="e.g., How many casual leaves do I get per year?"
              onKeyDown={e => e.key === "Enter" && onAsk()}
            />
            <div className="ctaRow">
              <button className="btn primary" disabled={!canAsk} onClick={onAsk}>
                {loading ? <span className="dots" /> : "Ask"}
              </button>
              <button className="btn" onClick={onClear}>Clear</button>
              <button className="btn ghost" onClick={copyAnswer} disabled={!res?.answer}>
                Copy Answer
              </button>
            </div>
          </div>
          <p className="hint">Keep it short for best results (e.g., “probation period”)</p>
        </section>

        {/* card: result */}
        {err && (
          <section className="card error">
            {err}
          </section>
        )}

        {res && (
          <section className="card result">
            <div className="resultHeader">
              <h2>Answer</h2>
              {res.confidence && (
                <span className={`badge ${res.confidence}`}>Confidence: {res.confidence}</span>
              )}
            </div>

            <p className="answer">{res.answer || "—"}</p>

            {/* policy matches */}
            {Array.isArray(res.policy_matches) && res.policy_matches.length > 0 && (
              <div className="chips">
                {res.policy_matches.map(m => (
                  <span key={m} className="chip">{m}</span>
                ))}
              </div>
            )}

            {/* citations */}
            {Array.isArray(res.citations) && res.citations.length > 0 && (
              <>
                <h3 className="mt">Citations</h3>
                <ul className="cites">
                  {res.citations.map((c, i) => (
                    <li key={i}>
                      <strong>{c.doc_id ?? c.source ?? "source"}</strong>
                      {c.section ? ` — ${c.section}` : ""}
                      {typeof c.page === "number" ? ` (p.${c.page})` : ""}
                      {c.snippet ? ` — ${c.snippet}` : ""}
                    </li>
                  ))}
                </ul>
              </>
            )}

            {/* follow-ups */}
            {Array.isArray(res.follow_up_suggestions) && res.follow_up_suggestions.length > 0 && (
              <>
                <h3 className="mt">Follow-ups</h3>
                <ul className="list">
                  {res.follow_up_suggestions.map((s, i) => (
                    <li key={i}>
                      <button
                        className="link"
                        onClick={() => { setQ(s); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                      >
                        {s}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </section>
        )}

        <footer className="footer">
          Can’t find an answer?{" "}
          <a className="link" href="mailto:hr@yourcompany.com?subject=Policy%20Question">
            Contact HR
          </a>
        </footer>
      </main>
    </div>
  );
}

/* ——— small toast (no deps) ——— */
function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 250);
  }, 1500);
}

/* ——— logo ——— */
function Logo() {
  return (
    <div className="logo" aria-hidden>
      <svg viewBox="0 0 48 48">
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#4f46e5" />
            <stop offset="1" stopColor="#06b6d4" />
          </linearGradient>
        </defs>
        <circle cx="24" cy="24" r="22" fill="url(#g)" opacity="0.15" />
        <path d="M10 30l8-16 6 10 4-6 10 12" fill="none" stroke="url(#g)" strokeWidth="3.2" strokeLinecap="round" />
      </svg>
    </div>
  );
}
